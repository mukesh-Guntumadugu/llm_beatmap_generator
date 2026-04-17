import os
import glob
import re
import argparse
import sqlite3
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import audio_feature_extraction as afe

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        print("Warning: Neither scikit-learn>=1.3 nor hdbscan is installed. HDBSCAN clustering will fail.")

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from prefixspan import PrefixSpan
except ImportError:
    print("Warning: prefixspan not installed. Sequential sequence mining will fail.")

def parse_ssc_sm(file_path):
    """
    Parses a single .ssc or .sm file and returns a list of dictionaries with difficulty and parsed measures.
    """
    metadata = {'bpms': None, 'offset': None, 'music': None}
    charts = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return charts, metadata

    # Extract High-level metadata
    bpms_match = re.search(r'#BPMS:\s*([^;]+);', content)
    offset_match = re.search(r'#OFFSET:\s*([^;]+);', content)
    music_match = re.search(r'#MUSIC:\s*([^;]+);', content)
    artist_match = re.search(r'#ARTIST:\s*([^;]+);', content)
    credit_match = re.search(r'#CREDIT:\s*([^;]+);', content)
    
    if bpms_match: metadata['bpms'] = bpms_match.group(1).strip()
    if offset_match: metadata['offset'] = offset_match.group(1).strip()
    if music_match: metadata['music'] = music_match.group(1).strip()
    
    metadata['author'] = "Unknown"
    credit_str = credit_match.group(1).strip() if credit_match else ""
    artist_str = artist_match.group(1).strip() if artist_match else ""
    if credit_str:
        metadata['author'] = credit_str
    elif artist_str:
        metadata['author'] = artist_str

    if file_path.endswith('.ssc'):
        sections = content.split('#NOTEDATA:')
        for sec in sections[1:]:
            stepstype_match = re.search(r'#STEPSTYPE:\s*([^;]+);', sec)
            if not stepstype_match or stepstype_match.group(1).strip() != 'dance-single':
                continue
            
            diff_match = re.search(r'#DIFFICULTY:\s*([^;]+);', sec)
            diff = diff_match.group(1).strip() if diff_match else "Unknown"
            
            notes_match = re.search(r'#NOTES:\s*\n*((?:[^;])+)', sec)
            if notes_match:
                notes_str = notes_match.group(1)
                charts.append({'difficulty': diff, 'notes_string': notes_str})
                
    elif file_path.endswith('.sm'):
        sections = content.split('#NOTES:')
        for sec in sections[1:]:
            parts = [p.strip() for p in sec.split(':')]
            if len(parts) >= 6:
                stepstype = parts[0]
                if stepstype != 'dance-single':
                    continue
                diff = parts[2]
                notes_str = parts[5].split(';')[0]
                charts.append({'difficulty': diff, 'notes_string': notes_str})
                
    return charts, metadata

def clean_and_split_measures(notes_str):
    """
    Splits the note string into individual measures and cleans comments/empty lines.
    Ensures lines have exactly 4 columns.
    """
    raw_measures = notes_str.split(',')
    cleaned_measures = []
    
    for raw_measure in raw_measures:
        lines = [line.strip() for line in raw_measure.split('\n')]
        # Filter comments and empty lines
        lines = [line for line in lines if line and not line.startswith('//')]
        
        # We enforce exactly 4 columns. If any line is != 4, we flag the measure as invalid.
        valid = True
        for line in lines:
            if len(line) != 4:
                valid = False
                break
                
        if valid and len(lines) > 0:
            cleaned_measures.append(lines)
            
    return cleaned_measures

def upscale_measure(lines, target_rows=192):
    """
    Projects varying-length measures onto a fixed 192 row grid.
    """
    num_lines = len(lines)
    output = [['0','0','0','0'] for _ in range(target_rows)]
    
    if num_lines == 0:
        return output
        
    for i, line in enumerate(lines):
        idx = int(i * target_rows / num_lines)
        if idx < target_rows:
            output[idx] = list(line)
            
    return output


# ---------------------------------------------------------------------------
# SYMMETRY CANONICALIZATION  (4-panel: L=col0  D=col1  U=col2  R=col3)
# ---------------------------------------------------------------------------
# Stepmania 4-panel layout on screen:  L  D  U  R
# Two useful symmetries that preserve rhythmic structure:
#   LR-mirror : Left↔Right AND Down↔Up  → reverse the 4-char string: abcd→dcba
#   UD-flip   : Down↔Up only            → swap positions 1&2:          abcd→acbd
# Combining gives 4 total transforms.  We pick the lexicographic minimum so
# that mirrored patterns (e.g. left-tap `1000` and right-tap `0001`) map to
# the same canonical form and cluster together in HDBSCAN.
# ---------------------------------------------------------------------------

_TRANSFORMS = [
    lambda a, b, c, d: (a, b, c, d),   # 0: identity
    lambda a, b, c, d: (d, c, b, a),   # 1: LR mirror  (L↔R, D↔U)
    lambda a, b, c, d: (a, c, b, d),   # 2: UD flip    (D↔U only)
    lambda a, b, c, d: (d, b, c, a),   # 3: LR + UD
]


def _apply_row(row, t_idx: int) -> tuple:
    """Apply transform t_idx to a single 4-element row (list or string)."""
    a, b, c, d = row[0], row[1], row[2], row[3]
    return _TRANSFORMS[t_idx](a, b, c, d)


def canonicalize_row(row) -> str:
    """
    Return the canonical (lexicographically smallest) single-row string
    across all 4 mirror/flip transforms.
    Used for sequence mining so mirrored step-strings hash identically.

    Examples
    --------
    '1000' (Left)  → min('1000','0001','1000','0001') = '0001'
    '0100' (Down)  → min('0100','0010','0010','0100') = '0010'
    Both Down and Up canonicalize to '0010'.
    """
    candidates = ["".join(_apply_row(row, t)) for t in range(4)]
    return min(candidates)


def canonicalize_measure(rows: list) -> list:
    """
    Return the canonical form of a full 192-row measure.
    Applies each of the 4 mirror/flip transforms consistently across every
    row and returns the lexicographically smallest variant as a list of
    4-char strings.

    This ensures that a measure of all left-taps and a measure of all
    right-taps (a perfect LR-mirror) map to the same encoding and are
    grouped into the same HDBSCAN cluster.
    """
    variants = [
        ["".join(_apply_row(row, t)) for row in rows]
        for t in range(4)
    ]
    # Python list comparison is element-by-element (lexicographic),
    # so min() returns the variant whose first differing row is smallest.
    return min(variants)


# ---------------------------------------------------------------------------
# NORMALIZATION STRATEGIES
# Three ways to force the computer to see physically-identical patterns as
# structurally identical, regardless of shift, stretch, or mirror.
# ---------------------------------------------------------------------------

def _active_cols(row) -> list:
    """Return list of column indices that are non-zero in a 4-char row."""
    return [i for i, ch in enumerate(row) if ch != '0']


# ── 1. Feature Extraction ───────────────────────────────────────────────────
# Instead of raw 0s/1s, compute 9 human-interpretable summary statistics
# per measure.  Two measures that 'look' the same to a human will have nearly
# identical trait vectors even if their raw matrices differ.

def is_hold_safe(chop_lines: list) -> bool:
    """Check if chop perfectly encapsulates holds without cutting them."""
    for c in range(4):
        col_chars = [row[c] for row in chop_lines if len(row) > c]
        if col_chars.count('2') != col_chars.count('3'):
            return False
    return True

def extract_measure_features(measure_lines: list, incoming_holds: int, outgoing_holds: int) -> list:
    """
    Return a 15-element feature vector for one measure or combined chop.

    Features
    --------
    0  total_active_steps   : number of non-zero rows (including holds)
    1  total_jumps          : rows where 2+ columns are simultaneously active
    2  max_col_distance     : largest column jump between consecutive single notes
    3  avg_col_distance     : average column movement (float, scaled 0-1)
    4  returns_to_start     : 1 if last active column == first active column
    5  step_density         : active_steps / total_rows  (0.0 – 1.0)
    6  unique_columns_used  : how many distinct columns appear (0-4)
    7  has_holds            : 1 if any hold character present
    8  has_mines            : 1 if any 'M' character present
    9  hold_duration        : ratio of rows with active holds
    10 symmetry_bias        : left/down bias vs up/right bias
    11 crossover_count      : count of physical crossovers
    12 incoming_holds       : active holds carried over from previous measure
    13 outgoing_holds       : active holds bridging into next measure
    14 longest_diff_chain   : length of longest sequence of identical relative movement (e.g. continuous staircase)
    """
    total_rows   = len(measure_lines)
    active_rows  = [r for r in measure_lines if r != '0000']
    total_active = len(active_rows)

    if total_active == 0:
        return [0] * 12 + [incoming_holds, outgoing_holds, 0]

    col_lists      = [_active_cols(r) for r in active_rows]
    jumps          = sum(1 for cols in col_lists if len(cols) >= 2)
    single_cols    = [cols[0] for cols in col_lists if len(cols) == 1]

    if len(single_cols) >= 2:
        dists       = [abs(single_cols[i+1] - single_cols[i]) for i in range(len(single_cols)-1)]
        max_dist    = max(dists)
        avg_dist    = sum(dists) / len(dists) / 3.0   # normalise to 0-1 (max possible = 3)
        returns     = int(single_cols[0] == single_cols[-1])
    else:
        max_dist = avg_dist = returns = 0

    density  = total_active / max(total_rows, 1)
    uniq_col = len({c for cols in col_lists for c in cols})
    has_hold = int(any(ch in ('2', '4', 'H') for r in active_rows for ch in r))
    has_mine = int(any('M' in r for r in active_rows))

    hold_duration = sum(1 for r in measure_lines for ch in r if ch in ('2', '3', '4', 'H')) / max(total_rows, 1)

    col_counts = [sum(1 for r in measure_lines if r[i] != '0') for i in range(4)]
    total_taps = sum(col_counts)
    symmetry_bias = (col_counts[0] + col_counts[1]) / max(total_taps, 1)

    crossover_count = 0
    if len(single_cols) >= 3:
        for i in range(len(single_cols)-2):
            seq = (single_cols[i], single_cols[i+1], single_cols[i+2])
            if seq in [(0,1,3), (3,1,0), (0,2,3), (3,2,0)]:
                crossover_count += 1

    longest_chain = 0
    if len(single_cols) >= 2:
        diff_chain = [((single_cols[i+1] - single_cols[i]) % 4) for i in range(len(single_cols)-1)]
        if diff_chain:
            current_streak = 1
            max_streak = 1
            for i in range(1, len(diff_chain)):
                if diff_chain[i] == diff_chain[i-1]:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
            longest_chain = max_streak

    return [total_active, jumps, max_dist, round(avg_dist, 4),
            returns, round(density, 4), uniq_col, has_hold, has_mine,
            round(hold_duration, 4), round(symmetry_bias, 4), crossover_count,
            incoming_holds, outgoing_holds, longest_chain]


# ── 2. Relative Encoding ────────────────────────────────────────────────────
# Encode the inter-step column *delta* rather than absolute position.
# Pattern A (L→D): col 0 → col 1 = delta +1
# Pattern B (D→U): col 1 → col 2 = delta +1
# Both become [+1] and map to the same cluster.

def encode_measure_relative(measure_lines: list) -> list:
    """
    Return a 7-element histogram of column deltas for a measure.
    Bins: delta = -3, -2, -1, 0, +1, +2, +3 (all relative movements possible
    on a 4-column pad).  Normalised by total transitions so measures of
    different lengths are comparable.

    Example
    -------
    L→D→U→R  =  deltas [+1, +1, +1]  →  bin[+1] = 3/3 = 1.0
    R→U→D→L  =  deltas [-1,-1,-1]    →  bin[-1] = 3/3 = 1.0  (same shape!)
    """
    active = [r for r in measure_lines if r != '0000']
    # Use first active column of each row (ignore jumps for delta calculation)
    single_cols = []
    for r in active:
        cols = _active_cols(r)
        if cols:                    # take the leftmost active column
            single_cols.append(cols[0])

    if len(single_cols) < 2:
        return [0.0] * 7            # not enough steps to compute deltas

    deltas    = [single_cols[i+1] - single_cols[i] for i in range(len(single_cols)-1)]
    bins      = {d: 0 for d in range(-3, 4)}
    for d in deltas:
        if d in bins:
            bins[d] += 1
    n         = len(deltas)
    return [bins[d] / n for d in range(-3, 4)]   # normalised counts


# ── 3. Run-Length Encoding (RLE) ────────────────────────────────────────────
# Stretch-invariant representation: a pattern played as 8th notes in a fast
# song and 4th notes in a slow song has identical RLE output.

def encode_measure_rle(measure_lines: list) -> list:
    """
    Compress a measure into (step, normalised_wait) events and return
    summary statistics as a fixed-length feature vector.

    Returns a 6-element vector:
      0  num_events          : number of compressed events (active steps)
      1  avg_wait_norm       : average normalised gap between steps (0-1)
      2  max_wait_norm       : largest gap (0-1)
      3  min_wait_norm       : smallest gap between steps (0-1)
      4  wait_std_norm       : std-dev of gaps (rhythm regularity)
      5  density             : num_events / total_rows (same as features[5])
    """
    total   = len(measure_lines)
    events  = []   # list of (canonical_step_str, wait_before)
    wait    = 0
    for r in measure_lines:
        if r == '0000':
            wait += 1
        else:
            events.append((canonicalize_row(r), wait))
            wait = 0

    n = len(events)
    if n == 0:
        return [0.0] * 6

    waits_norm = [e[1] / max(total, 1) for e in events]
    avg_w  = sum(waits_norm) / n
    max_w  = max(waits_norm)
    min_w  = min(waits_norm)
    variance = sum((w - avg_w) ** 2 for w in waits_norm) / n
    std_w  = variance ** 0.5

    return [n, round(avg_w, 4), round(max_w, 4),
            round(min_w, 4), round(std_w, 4), round(n / max(total, 1), 4)]


# ---------------------------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------------------------

def init_database(db_path: str) -> sqlite3.Connection:
    """
    Creates (or opens) the SQLite database and ensures ALL tables exist.

    Tables:
      processed_files        — one row per .sm/.ssc file scanned
      difficulty_breakdown   — per-difficulty 4-col stats per run
      character_distributions — character frequency counts per run
      cluster_counts         — HDBSCAN cluster sizes per run
      markov_transitions     — Markov chain transition probs per run
      prefixspan_patterns    — top PrefixSpan sequential patterns per run

    Cluster-safe settings:
      - WAL journal mode  → parallel SLURM jobs can read simultaneously
      - busy_timeout 30 s → writer queues instead of crashing on NFS/Lustre
    """
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA busy_timeout = 30000;")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS processed_files (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path           TEXT    UNIQUE NOT NULL,
            song_name           TEXT    NOT NULL,
            format              TEXT    NOT NULL,
            author              TEXT,
            skipped_duplicate   INTEGER NOT NULL DEFAULT 0,
            difficulties_found  TEXT,
            measures_4col       INTEGER DEFAULT 0,
            processed_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS duplicate_file_name (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path           TEXT    NOT NULL,
            song_name           TEXT    NOT NULL,
            format              TEXT    NOT NULL,
            author              TEXT,
            is_same_name        INTEGER,
            processed_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS difficulty_breakdown (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                  TEXT    NOT NULL,
            difficulty              TEXT    NOT NULL,
            total_charts_in_files   INTEGER,
            charts_with_4col        INTEGER,
            total_4col_measures     INTEGER,
            skipped_charts_non4col  INTEGER,
            saved_at                TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS character_distributions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            character       TEXT    NOT NULL,
            difficulty      TEXT    NOT NULL,
            count           INTEGER,
            is_new_pattern  TEXT,
            saved_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_counts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT    NOT NULL,
            cluster_id  INTEGER,
            count       INTEGER,
            saved_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS markov_transitions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            current_state   TEXT    NOT NULL,
            next_state      TEXT    NOT NULL,
            count           INTEGER,
            probability     REAL,
            saved_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prefixspan_patterns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT    NOT NULL,
            pattern     TEXT    NOT NULL,
            frequency   INTEGER,
            saved_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_transitions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            current_cluster INTEGER NOT NULL,
            next_cluster    INTEGER NOT NULL,
            count           INTEGER,
            probability     REAL,
            saved_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audio_features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            file_path       TEXT    NOT NULL,
            difficulty      TEXT    NOT NULL,
            measure_idx     INTEGER NOT NULL,
            chop_length     INTEGER DEFAULT 1,
            start_time      REAL,
            end_time        REAL,
            rms_energy      REAL,
            onset_density   REAL,
            tempo_strength  REAL,
            chroma_mean     REAL,
            spectral_centroid REAL,
            spectral_bandwidth REAL,
            spectral_contrast REAL,
            spectral_flatness REAL,
            vocal_word_count  REAL,
            vocal_density     REAL,
            mfcc_0 REAL, mfcc_1 REAL, mfcc_2 REAL, mfcc_3 REAL, mfcc_4 REAL, mfcc_5 REAL, 
            mfcc_6 REAL, mfcc_7 REAL, mfcc_8 REAL, mfcc_9 REAL, mfcc_10 REAL, mfcc_11 REAL, mfcc_12 REAL,
            saved_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stepmania_features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            file_path       TEXT    NOT NULL,
            difficulty      TEXT    NOT NULL,
            measure_idx     INTEGER NOT NULL,
            chop_length     INTEGER DEFAULT 1,
            total_active    INTEGER,
            jumps           INTEGER,
            max_dist        INTEGER,
            avg_dist        REAL,
            returns         INTEGER,
            density         REAL,
            uniq_col        INTEGER,
            has_hold        INTEGER,
            has_mine        INTEGER,
            hold_duration   REAL,
            symmetry_bias   REAL,
            crossover_count INTEGER,
            incoming_holds  INTEGER,
            outgoing_holds  INTEGER,
            longest_diff_chain INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS measure_cluster_assignments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            file_path       TEXT    NOT NULL,
            difficulty      TEXT    NOT NULL,
            measure_idx     INTEGER NOT NULL,
            chop_length     INTEGER DEFAULT 1,
            cluster_id      INTEGER NOT NULL,
            saved_at        TEXT    NOT NULL
        );
    """)
    try:
        conn.execute("ALTER TABLE processed_files ADD COLUMN author TEXT;")
    except sqlite3.OperationalError:
        pass
    for col_def in [
        ("stepmania_features", "incoming_holds INTEGER"),
        ("stepmania_features", "outgoing_holds INTEGER"),
        ("stepmania_features", "chop_length INTEGER DEFAULT 1"),
        ("stepmania_features", "longest_diff_chain INTEGER DEFAULT 0"),
        ("audio_features", "chop_length INTEGER DEFAULT 1"),
        ("measure_cluster_assignments", "chop_length INTEGER DEFAULT 1"),
    ]:
        table, definition = col_def
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition};")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def log_file_to_db(conn: sqlite3.Connection, file_path: str, song_name: str,
                   fmt: str, skipped: bool,
                   difficulties: list = None, measures_4col: int = 0,
                   author: str = "Unknown",
                   max_retries: int = 5):
    """
    Inserts or replaces a file record in the DB.
    Call with skipped=True to record a duplicate that was not processed.

    Retries up to max_retries times with exponential back-off so that
    parallel SLURM jobs writing to the same shared DB don't crash on
    transient lock errors common on NFS-mounted cluster storage.
    """
    import time, random
    diff_str = ",".join(sorted(difficulties)) if difficulties else ""
    sql = """
        INSERT OR REPLACE INTO processed_files
            (file_path, song_name, format, author, skipped_duplicate,
             difficulties_found, measures_4col, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        file_path, song_name, fmt, author,
        1 if skipped else 0,
        diff_str, measures_4col,
        datetime.now().isoformat()
    )
    for attempt in range(max_retries):
        try:
            conn.execute(sql, params)
            conn.commit()
            return                          # success — exit immediately
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()   # 1s, 2s, 4s, 8s …
                print(f"  [DB] Write locked, retrying in {wait:.1f}s "
                      f"(attempt {attempt+1}/{max_retries})…")
                time.sleep(wait)
            else:
                raise   # give up after max_retries or non-lock error

def log_duplicate_to_db(conn: sqlite3.Connection, file_path: str, song_name: str,
                        fmt: str, author: str, is_same_name: int,
                        max_retries: int = 5):
    import time, random
    sql = """
        INSERT INTO duplicate_file_name
            (file_path, song_name, format, author, is_same_name, processed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (file_path, song_name, fmt, author, is_same_name, datetime.now().isoformat())
    for attempt in range(max_retries):
        try:
            conn.execute(sql, params)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# MAIN DATASET PROCESSING
# ---------------------------------------------------------------------------

def process_dataset(directory, db_path, run_id, chop_mode='both'):
    print(f"Scanning directory: {directory} for .ssc and .sm files...")
    search_pattern_ssc = os.path.join(directory, '**', '*.ssc')
    search_pattern_sm  = os.path.join(directory, '**', '*.sm')

    all_files = sorted(
        glob.glob(search_pattern_ssc, recursive=True) +
        glob.glob(search_pattern_sm,  recursive=True)
    )

    # ------------------------------------------------------------------
    # DEDUPLICATION: prefer .ssc over .sm when the same base name exists
    # in the same directory (e.g. "Goin' Under.sm" + "Goin' Under.ssc").
    # Group by (parent_dir, stem) and keep the .ssc if both are present.
    # ------------------------------------------------------------------
    conn = init_database(db_path)

    seen_keys   = {}   # (parent_dir, stem) -> chosen file path
    skipped_map = {}   # file_path -> True  (will be logged to DB as skipped)

    for fp in all_files:
        parent = os.path.dirname(fp)
        stem   = os.path.splitext(os.path.basename(fp))[0]
        key    = (parent, stem)
        ext    = os.path.splitext(fp)[1].lower()   # '.sm' or '.ssc'

        if key not in seen_keys:
            seen_keys[key] = fp
        else:
            existing_ext = os.path.splitext(seen_keys[key])[1].lower()
            if ext == '.ssc':               # .ssc wins — swap in, mark old as skipped
                skipped_map[seen_keys[key]] = True
                seen_keys[key] = fp
            else:                           # .sm loses — mark current as skipped
                skipped_map[fp] = True

    files_to_process = sorted(seen_keys.values())
    files_skipped    = sorted(skipped_map.keys())
    total            = len(files_to_process)
    total_raw        = len(all_files)

    # Log every skipped duplicate to the DB immediately
    for fp in files_skipped:
        _, metadata = parse_ssc_sm(fp)
        song_name = os.path.splitext(os.path.basename(fp))[0]
        fmt = os.path.splitext(fp)[1]
        author = metadata.get('author', 'Unknown')
        
        log_file_to_db(conn, fp,
                       song_name=song_name,
                       fmt=fmt,
                       author=author,
                       skipped=True)
                       
        log_duplicate_to_db(conn, fp,
                            song_name=song_name,
                            fmt=fmt,
                            author=author,
                            is_same_name=1)

    print(f"Found {total_raw} raw files → {len(files_skipped)} duplicates skipped → {total} to process.\n")

    # Print full list split into two sections
    print("=" * 60)
    print(f"  FILES TO PROCESS ({total} total — duplicates removed)")
    print("=" * 60)
    for i, fp in enumerate(files_to_process, 1):
        print(f"  [{i:>3}/{total}] {os.path.relpath(fp, directory)}")

    if files_skipped:
        print()
        print(f"  SKIPPED DUPLICATES ({len(files_skipped)} — .ssc preferred over .sm)")
        print("-" * 60)
        for fp in files_skipped:
            print(f"  [SKIP] {os.path.relpath(fp, directory)}")
    print("=" * 60)
    print()

    char_counts = Counter()

    # Detailed per-difficulty tracking for the breakdown CSV
    # Structure: { difficulty: { 'total_charts': int, 'files_with_4col': int, 'total_4col_measures': int } }
    diff_stats = {}

    all_raw_measures      = []   # Kept for DB insertion sequence
    all_sequence_measures = []   # For PrefixSpan/Markov
    all_measure_info      = []   # Links logic: (file_path, difficulty, measure_idx)

    for count, fp in enumerate(files_to_process):
        print(f"  Processing [{count+1:>3}/{total}]: {os.path.basename(fp)}")
        if count > 0 and count % 100 == 0:
            print(f"  --- Checkpoint: {count}/{total} files processed ---")

        charts, metadata = parse_ssc_sm(fp)
        
        # Audio feature setup
        y, sr, time_map, vocal_words = None, None, None, []
        if metadata['music']:
            audio_path = os.path.join(os.path.dirname(fp), metadata['music'])
        else:
            base = os.path.join(os.path.dirname(fp), os.path.splitext(os.path.basename(fp))[0])
            if os.path.exists(base+'.ogg'): audio_path = base+'.ogg'
            elif os.path.exists(base+'.mp3'): audio_path = base+'.mp3'
            else: audio_path = None
            
        if audio_path:
            y, sr = afe.load_audio(audio_path)
            time_map = afe.parse_time_map(metadata['bpms'], metadata['offset'])
            vocal_words = afe.transcribe_vocals(audio_path)
            
        audio_rows = []
        stepmania_rows = []

        for chart in charts:
            diff = chart['difficulty']

            # Initialise stats bucket for this difficulty if first time seen
            if diff not in diff_stats:
                diff_stats[diff] = {'total_charts': 0, 'files_with_4col': 0, 'total_4col_measures': 0}

            diff_stats[diff]['total_charts'] += 1

            # Only process 4-column (dance-single) measures — others are silently skipped
            cleaned_measures = clean_and_split_measures(chart['notes_string'])

            if cleaned_measures:  # This chart had at least one valid 4-col measure
                diff_stats[diff]['files_with_4col'] += 1
                diff_stats[diff]['total_4col_measures'] += len(cleaned_measures)

            chart_interpolated = []
            chart_raw = []
            active_holds = [False] * 4
            
            for measure_lines in cleaned_measures:
                chart_raw.append(measure_lines)
                interpolated_measure = []
                for line in measure_lines:
                    new_line = ""
                    for c, char in enumerate(line):
                        if char in ('2', '4'):
                            active_holds[c] = True
                            new_line += char
                        elif char == '3':
                            active_holds[c] = False
                            new_line += char
                        elif char == '0' and active_holds[c]:
                            new_line += 'H'
                        else:
                            new_line += char
                    interpolated_measure.append(new_line)
                chart_interpolated.append(interpolated_measure)

            total_m = len(cleaned_measures)
            chops_to_process = []
            
            if chop_mode in ['sliding', 'both']:
                for start_idx in range(total_m):
                    for length in [1, 2, 3]:
                        end_idx = start_idx + length
                        if end_idx <= total_m:
                            chop_raw = [line for m in chart_raw[start_idx:end_idx] for line in m]
                            chop_interp = [line for m in chart_interpolated[start_idx:end_idx] for line in m]
                            if is_hold_safe(chop_raw):
                                chops_to_process.append((start_idx, length, chop_raw, chop_interp))
                                
            if chop_mode in ['discrete', 'both']:
                start_idx = 0
                while start_idx < total_m:
                    length = 1
                    while start_idx + length <= total_m:
                        chop_raw = [line for m in chart_raw[start_idx:start_idx+length] for line in m]
                        if is_hold_safe(chop_raw) or (start_idx + length == total_m):
                            break
                        length += 1
                        if length > 8:
                            break
                    if start_idx + length <= total_m:
                        chop_raw = [line for m in chart_raw[start_idx:start_idx+length] for line in m]
                        chop_interp = [line for m in chart_interpolated[start_idx:start_idx+length] for line in m]
                        chop_tuple = (start_idx, length, chop_raw, chop_interp)
                        if chop_tuple not in chops_to_process:
                             chops_to_process.append(chop_tuple)
                    start_idx += length

            for start_idx, chop_len, raw_lines, interp_lines in chops_to_process:
                # Count characters by difficulty using the interpolated lines (only for length=1 to avoid massive duplication in tables)
                if chop_len == 1:
                    for line in interp_lines:
                        for char in line:
                            char_counts[(char, diff)] += 1

                all_raw_measures.append(raw_lines)
                all_measure_info.append((fp, diff, start_idx, chop_len))

                active_steps = [canonicalize_row(line) for line in raw_lines if line != '0000']
                if active_steps:
                    all_sequence_measures.append(active_steps)
                    
                if y is not None and time_map is not None:
                    start_beat = start_idx * 4.0
                    end_beat = (start_idx + chop_len) * 4.0
                    start_time = afe.get_time_for_beat(start_beat, time_map)
                    end_time = afe.get_time_for_beat(end_beat, time_map)
                    feats = afe.extract_audio_features_for_slice(y, sr, start_time, end_time)
                    vocal_feats = afe.get_vocal_features_for_slice(vocal_words, start_time, end_time)
                    
                    row_time = datetime.now().isoformat()
                    audio_rows.append((run_id, fp, diff, start_idx, chop_len, start_time, end_time) + tuple(feats) + tuple(vocal_feats) + (row_time,))

                incoming_holds = 0
                outgoing_holds = 0
                if len(interp_lines) > 0:
                    incoming_holds = interp_lines[0].count('H')
                    outgoing_holds = interp_lines[-1].count('H') + interp_lines[-1].count('2')

                sm_feats = extract_measure_features(interp_lines, incoming_holds, outgoing_holds)
                stepmania_rows.append((run_id, fp, diff, start_idx, chop_len) + tuple(sm_feats))
                
        if audio_rows:
            conn.executemany(
                "INSERT INTO audio_features (run_id, file_path, difficulty, measure_idx, chop_length, start_time, end_time, rms_energy, onset_density, tempo_strength, chroma_mean, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, mfcc_0, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, vocal_word_count, vocal_density, saved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                audio_rows
            )
            
        if stepmania_rows:
            conn.executemany(
                "INSERT INTO stepmania_features (run_id, file_path, difficulty, measure_idx, chop_length, total_active, jumps, max_dist, avg_dist, returns, density, uniq_col, has_hold, has_mine, hold_duration, symmetry_bias, crossover_count, incoming_holds, outgoing_holds, longest_diff_chain) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                stepmania_rows
            )
        conn.commit()
        
        # Explicitly free memory footprint (Audio array & Whisper VRAM)
        if 'y' in locals() and y is not None:
            del y
            del time_map
            del vocal_words
            
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        file_diffs    = [c['difficulty'] for c in charts]
        file_measures = sum(
            len(clean_and_split_measures(c['notes_string'])) for c in charts
        )
        log_file_to_db(conn,
                       file_path=fp,
                       song_name=os.path.splitext(os.path.basename(fp))[0],
                       fmt=os.path.splitext(fp)[1],
                       skipped=False,
                       difficulties=file_diffs,
                       measures_4col=file_measures,
                       author=metadata.get('author', 'Unknown'))

    conn.close()
    print(f"\\nDatabase updated: {db_path}")
    print("Data parsing complete!")
    return all_raw_measures, all_sequence_measures, all_measure_info, diff_stats, char_counts

def run_topology_pipeline(raw_measures, measure_info, conn, run_id, output_dir, encoding='features'):
    """
    encoding choices:
      'ohe'      – One-Hot Encoding of canonical 192x4 matrix  (default / most detailed)
      'features' – 9-trait feature extraction  (ultimate solution, physically-invariant)
      'relative' – Column-delta histogram       (shift-invariant)
      'rle'      – Run-length compression stats (stretch-invariant)
    """
    print(f"\n--- Topology Pipeline  [encoding: {encoding}] ---")
    if not measure_info:
        print("No valid measures to process.")
        return

    print(f"Total measures to cluster: {len(measure_info)}")

    # ── Select and build the encoding matrix ─────────────────────────────
    if encoding == 'features':
        print("Extracting physical trait feature vectors directly from SQLite...")
        query = """
            SELECT 
                total_active, jumps, max_dist, avg_dist, returns, density, 
                uniq_col, has_hold, has_mine, hold_duration, symmetry_bias, 
                crossover_count, incoming_holds, outgoing_holds, longest_diff_chain
            FROM stepmania_features
            WHERE run_id = ?
            ORDER BY id ASC
        """
        df_feats = pd.read_sql(query, conn, params=(run_id,))
        encoded_data = df_feats.values
        print(f"Feature matrix shape: {encoded_data.shape}  (measures × 15 traits)")

    elif encoding == 'features+audio':
        print("Extracting physical and audio features with INNER JOIN...")
        query = """
            SELECT 
                s.file_path, s.difficulty, s.measure_idx, s.chop_length,
                s.total_active, s.jumps, s.max_dist, s.avg_dist, s.returns, s.density, 
                s.uniq_col, s.has_hold, s.has_mine, s.hold_duration, s.symmetry_bias, 
                s.crossover_count, s.incoming_holds, s.outgoing_holds, s.longest_diff_chain,
                a.rms_energy, a.onset_density, a.tempo_strength, a.spectral_centroid, 
                a.spectral_bandwidth, a.vocal_density,
                a.spectral_contrast, a.spectral_flatness,
                a.mfcc_0, a.mfcc_1, a.mfcc_2, a.mfcc_3, a.mfcc_4, a.mfcc_5, a.mfcc_6, 
                a.mfcc_7, a.mfcc_8, a.mfcc_9, a.mfcc_10, a.mfcc_11, a.mfcc_12
            FROM stepmania_features s
            INNER JOIN audio_features a
                ON s.run_id = a.run_id
                AND s.file_path = a.file_path
                AND s.difficulty = a.difficulty
                AND s.measure_idx = a.measure_idx
                AND s.chop_length = a.chop_length
            WHERE s.run_id = ?
            ORDER BY s.id ASC
        """
        df_feats = pd.read_sql(query, conn, params=(run_id,))
        
        # We need to filter raw_measures and measure_info to match the INNER JOIN result exactly.
        raw_map = {(mi[0], mi[1], int(mi[2]), int(mi[3])): rm for mi, rm in zip(measure_info, raw_measures)}
        keys_df = df_feats[['file_path', 'difficulty', 'measure_idx', 'chop_length']]
        
        new_measure_info = []
        new_raw_measures = []
        for fp, diff, midx, chplen in keys_df.itertuples(index=False, name=None):
            key = (fp, diff, int(midx), int(chplen))
            new_measure_info.append(key)
            new_raw_measures.append(raw_map.get(key, []))
            
        measure_info = new_measure_info
        raw_measures = new_raw_measures
        
        df_feats = df_feats.drop(columns=['file_path', 'difficulty', 'measure_idx', 'chop_length'])
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_feats.values)
        
        # Weight multiplier: Physical (idx 0 to 14) vs Audio (idx 15 to 35)
        weight_multiplier = 1.5
        scaled_data[:, :15] *= weight_multiplier
        
        encoded_data = scaled_data
        print(f"Features+Audio matrix shape: {encoded_data.shape}  (measures × 36 traits)")

    elif encoding == 'relative':
        print("Building column-delta histograms (Relative Encoding)...")
        encoded_data = np.array([encode_measure_relative(m) for m in raw_measures], dtype=float)
        print(f"Relative matrix shape: {encoded_data.shape}  (measures × 7 delta bins)")

    elif encoding == 'rle':
        print("Computing run-length compression stats (RLE Encoding)...")
        encoded_data = np.array([encode_measure_rle(m) for m in raw_measures], dtype=float)
        print(f"RLE matrix shape: {encoded_data.shape}  (measures × 6 timing stats)")

    else:
        print("WARNING: OHE encoding is deprecated due to massive memory requirements. Failsafe to 'features'.")
        # Run recursive fallback
        return run_topology_pipeline(raw_measures, measure_info, conn, run_id, output_dir, encoding='features')

    print(f"Encoded Shape: {encoded_data.shape}")
    # ── PCA → HDBSCAN → UMAP (same for all encodings) ───────────────────
    # For low-dimensional encodings (features=15, relative=7, rle=6),
    # PCA is a no-op or very light — that's fine.
    base_components = 24 if encoding == 'features+audio' else 16
    components = min(base_components, encoded_data.shape[1], encoded_data.shape[0])
    print(f"Compressing data heavily using PCA to {components} dimensions...")
    pca = PCA(n_components=components)
    compressed_data = pca.fit_transform(encoded_data)

    # 3. HDBSCAN
    print("Clustering dense neighborhoods with HDBSCAN...")
    hdbscan_clusterer = HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = hdbscan_clusterer.fit_predict(compressed_data)

    unique_clusters = set(labels)
    noise_points = list(labels).count(-1)
    print(f"HDBSCAN discovered {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} distinct pattern groups.")
    print(f"Filtered out {noise_points} messy noise measures (-1).")

    # Save cluster counts → DB
    now = datetime.now().isoformat()
    cluster_series = pd.Series(labels).value_counts()
    conn.executemany(
        "INSERT INTO cluster_counts (run_id, cluster_id, count, saved_at) VALUES (?,?,?,?)",
        [(run_id, int(cid), int(cnt), now) for cid, cnt in cluster_series.items()]
    )
    conn.commit()
    print(f"Cluster counts saved to DB (table: cluster_counts, run_id={run_id})")

    # Save measure cluster assignments -> DB
    print("Saving measure-to-cluster mapping...")
    assign_rows = []
    for i in range(len(labels)):
        # Provide all 4 items logic
        item = measure_info[i]
        fp, diff, midx = item[0], item[1], item[2]
        chop_len = item[3] if len(item) > 3 else 1
        assign_rows.append((run_id, fp, diff, midx, chop_len, int(labels[i]), now))
    
    conn.executemany(
        "INSERT INTO measure_cluster_assignments (run_id, file_path, difficulty, measure_idx, chop_length, cluster_id, saved_at) VALUES (?,?,?,?,?,?,?)",
        assign_rows
    )
    conn.commit()
    print(f"Measure mappings saved to DB (table: measure_cluster_assignments, run_id={run_id})")

    # Save representative examples of each cluster to a text file
    print("Exporting representative examples of clusters to markdown...")
    examples_path = os.path.join(output_dir, f'cluster_examples_{encoding}.md')
    try:
        with open(examples_path, 'w') as f:
            f.write(f"# Cluster Examples (Top 20 by size)\n\n")
            top_clusters = cluster_series.head(20)
            for cid, count in top_clusters.items():
                f.write(f"## Cluster {cid} (Size: {count})\n")
                f.write("```\n")
                
                # Find indices of measures that belong to this cluster
                indices = np.where(labels == cid)[0]
                # Pick up to 5 examples
                sampled_indices = indices[:5]
                
                for i, idx in enumerate(sampled_indices):
                    f.write(f"--- Example {i+1} ---\n")
                    f.write("\n".join(raw_measures[idx]))
                    f.write("\n\n")
                f.write("```\n\n")
        print(f"Cluster examples visually exported to {examples_path}")
    except Exception as e:
        print(f"Failed to export cluster examples: {e}")

    # Save Macro-Markov Chain (Cluster Transitions) → DB
    print("Calculating Macro-Markov Chain (Measure-level Cluster Transitions)...")
    cluster_trans = Counter()
    cluster_state_counts = Counter()
    for i in range(len(labels) - 1):
        cur, nxt = labels[i], labels[i+1]
        cluster_trans[(cur, nxt)] += 1
        cluster_state_counts[cur] += 1

    trans_rows = []
    for (cur, nxt), cnt in cluster_trans.items():
        prob = cnt / cluster_state_counts[cur]
        trans_rows.append((run_id, int(cur), int(nxt), int(cnt), round(prob, 4), now))

    conn.executemany(
        "INSERT INTO cluster_transitions "
        "(run_id, current_cluster, next_cluster, count, probability, saved_at) "
        "VALUES (?,?,?,?,?,?)",
        trans_rows
    )
    conn.commit()
    print(f"Cluster transitions saved to DB (table: cluster_transitions, run_id={run_id})")

    # 4. UMAP — still saved as PNG (images don't belong in a DB)
    print("Projecting clusters to 2D Map using UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(compressed_data)

    plt.figure(figsize=(12, 10))
    noise    = labels == -1
    clustered = labels != -1
    plt.scatter(embedding[noise, 0],     embedding[noise, 1],     c='lightgrey', s=5,  alpha=0.5, label='Noise (-1)')
    scatter = plt.scatter(embedding[clustered, 0], embedding[clustered, 1], c=labels[clustered], cmap='Spectral', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'UMAP — Stepmania Measures  [encoding: {encoding}]')
    plt.legend()
    png_path = os.path.join(output_dir, f'measure_cluster_map_{encoding}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {png_path}")
    plt.close()

    # 5. Feature Representation Heatmap (What traits does each cluster have?)
    if encoding in ['features', 'features+audio', 'relative', 'rle']:
        print("Generating Feature Profile Heatmap...")
        df_encoded = pd.DataFrame(encoded_data)
        df_encoded['cluster'] = labels
        # Exclude noise for the profile heatmap so we just see the clean patterns
        cluster_means = df_encoded[df_encoded['cluster'] != -1].groupby('cluster').mean()
        
        if encoding == 'features':
            col_names = ['active_steps', 'jumps', 'max_dist', 'avg_dist', 
                         'returns', 'density', 'uniq_cols', 'has_holds', 'has_mines',
                         'hold_duration', 'symmetry_bias', 'crossovers', 'in_holds', 'out_holds', 'longest_diff']
        elif encoding == 'features+audio':
            col_names = ['active_steps', 'jumps', 'max_dist', 'avg_dist', 'returns', 'density', 'uniq_cols', 'has_holds', 'has_mines', 'hold_duration', 'symmetry_bias', 'crossovers', 'in_holds', 'out_holds', 'longest_diff', 'rms_energy', 'onset_density', 'tempo_strength', 'spectral_centroid', 'spectral_bandwidth', 'vocal_density', 'spectral_contrast', 'spectral_flatness', 'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12']
        elif encoding == 'relative':
            col_names = ['-3', '-2', '-1', '0', '+1', '+2', '+3']
        elif encoding == 'rle':
            col_names = ['num_events', 'avg_wait', 'max_wait', 'min_wait', 'wait_std', 'density']
        else:
            col_names = [str(i) for i in range(encoded_data.shape[1])]
            
        cluster_means.columns = col_names
        
        plt.figure(figsize=(10, max(4, len(cluster_means) * 0.5 + 2)))
        sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Average Features per Cluster [{encoding}]")
        plt.ylabel("Cluster ID")
        feat_heat_path = os.path.join(output_dir, f'feature_heatmap_{encoding}.png')
        plt.savefig(feat_heat_path, dpi=300, bbox_inches='tight')
        print(f"Feature Heatmap saved to {feat_heat_path}")
        plt.close()

    # 6. Macro-Markov Heatmap (Cluster Transition Matrix)
    print("Generating Macro-Markov (Cluster Transition) Heatmap...")
    if len(trans_rows) > 0:
        df_trans = pd.DataFrame(trans_rows, columns=['run_id', 'cur', 'nxt', 'count', 'prob', 'saved_at'])
        pivot_trans = df_trans.pivot(index='cur', columns='nxt', values='prob').fillna(0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_trans, annot=True, cmap="Reds", fmt=".2f")
        plt.title("Macro-Markov Transition Probabilities (Cluster -> Cluster)")
        plt.xlabel("Next Cluster")
        plt.ylabel("Current Cluster")
        macro_heat_path = os.path.join(output_dir, 'macro_cluster_transitions_heatmap.png')
        plt.savefig(macro_heat_path, dpi=300, bbox_inches='tight')
        print(f"Macro-Markov Heatmap saved to {macro_heat_path}")
        plt.close()


def run_sequential_pipeline(sequence_measures, conn, run_id, output_dir):
    print(f"\n--- Sequential Pattern Mining ---")
    if not sequence_measures:
        return

    now = datetime.now().isoformat()

    # Markov Chain
    print("Calculating Markov Chain State Transitions...")
    transitions  = Counter()
    state_counts = Counter()
    for seq in sequence_measures:
        for i in range(len(seq) - 1):
            transitions[(seq[i], seq[i+1])] += 1
            state_counts[seq[i]] += 1

    markov_rows = []
    for (cur, nxt), cnt in transitions.items():
        prob = cnt / state_counts[cur]
        markov_rows.append((run_id, cur, nxt, cnt, round(prob, 4), now))

    conn.executemany(
        "INSERT INTO markov_transitions "
        "(run_id, current_state, next_state, count, probability, saved_at) "
        "VALUES (?,?,?,?,?,?)",
        markov_rows
    )
    conn.commit()
    print(f"Saved {len(markov_rows)} Markov transitions to DB (table: markov_transitions, run_id={run_id})")

    # Generate Micro-Markov Heatmap
    print("Generating Micro-Markov Heatmap for Top 15 States...")
    if len(markov_rows) > 0:
        df_micro = pd.DataFrame(markov_rows, columns=['run_id', 'cur', 'nxt', 'count', 'prob', 'saved_at'])
        # Get top 15 most frequent states to keep heatmap readable
        top_states = [x[0] for x in state_counts.most_common(15)]
        
        df_top = df_micro[(df_micro['cur'].isin(top_states)) & (df_micro['nxt'].isin(top_states))]
        
        if not df_top.empty:
            pivot_micro = df_top.pivot(index='cur', columns='nxt', values='prob').fillna(0)
            pivot_micro = pivot_micro.reindex(index=top_states, columns=top_states, fill_value=0)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_micro, annot=True, cmap="Blues", fmt=".2f")
            plt.title("Micro-Markov Transition Probabilities (Top 15 States)")
            plt.xlabel("Next Step State")
            plt.ylabel("Current Step State")
            micro_heat_path = os.path.join(output_dir, 'micro_markov_transitions_heatmap.png')
            plt.savefig(micro_heat_path, dpi=300, bbox_inches='tight')
            print(f"Micro-Markov Heatmap saved to {micro_heat_path}")
            plt.close()

    # PrefixSpan
    print("Mining frequent sequential sub-patterns using PrefixSpan...")
    try:
        ps = PrefixSpan(sequence_measures)
        min_support     = max(10, len(sequence_measures) // 100)
        frequent_patterns = ps.frequent(min_support)
        sub_patterns    = sorted(
            [p for p in frequent_patterns if len(p[1]) > 1],
            key=lambda x: x[0], reverse=True
        )[:100]

        ps_rows = [
            (run_id, " -> ".join(pattern), cnt, now)
            for cnt, pattern in sub_patterns
        ]
        conn.executemany(
            "INSERT INTO prefixspan_patterns (run_id, pattern, frequency, saved_at) VALUES (?,?,?,?)",
            ps_rows
        )
        conn.commit()
        print(f"Saved {len(ps_rows)} PrefixSpan patterns to DB (table: prefixspan_patterns, run_id={run_id})")
    except Exception as e:
        print(f"PrefixSpan mining failed: {e}")

def generate_audio_cluster_correlations(conn, run_id, output_dir):
    print("\n--- Generating Audio-Cluster Correlations ---")
    query = """
    SELECT 
        c.cluster_id, 
        a.rms_energy, 
        a.onset_density, 
        a.tempo_strength, 
        a.spectral_centroid, 
        a.vocal_density,
        a.spectral_contrast,
        a.spectral_flatness
    FROM measure_cluster_assignments c
    JOIN audio_features a 
      ON c.run_id = a.run_id 
     AND c.file_path = a.file_path 
     AND c.difficulty = a.difficulty 
     AND c.measure_idx = a.measure_idx
    WHERE c.cluster_id != -1 AND c.run_id = ?
    """
    
    try:
        df = pd.read_sql(query, conn, params=(run_id,))
    except Exception as e:
        print(f"Failed to query correlations: {e}")
        return
        
    if df.empty:
        print("No correlation data found. Skipping audio-cluster visualizations.")
        return
        
    print(f"Successfully joined {len(df):,} beatmap measures across both musical audio and physical patterns!")
    
    # 1. Filter to top 20 largest clusters
    top_clusters = df['cluster_id'].value_counts().nlargest(20).index
    df_top = df[df['cluster_id'].isin(top_clusters)]
    
    if df_top.empty:
        return

    # 2. Average audio features per cluster
    cluster_means = df_top.groupby('cluster_id').mean()
    
    # Safely Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_means)
    df_scaled = pd.DataFrame(scaled_features, index=cluster_means.index, columns=cluster_means.columns)
    
    print("Generating Audio Mood Heatmap...")
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_scaled, cmap='coolwarm', center=0, annot=True, fmt=".2f")
    plt.title(f"Audio Mood Profiles for Top 20 Stepmania Clusters (run: {run_id})")
    plt.ylabel("HDBSCAN Cluster ID")
    plt.xlabel("Audio Characteristic")
    
    heat_path = os.path.join(output_dir, 'audio_mood_heatmap.png')
    plt.savefig(heat_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating Vocal Distribution Boxplots...")
    plt.figure(figsize=(14, 6))
    top_8_clusters = df['cluster_id'].value_counts().nlargest(8).index
    df_8 = df[df['cluster_id'].isin(top_8_clusters)]
    if not df_8.empty:
        sns.boxplot(data=df_8, x='cluster_id', y='vocal_density', palette='plasma')
        plt.title("Vocal Presence Spread Across Top 8 Physical Clusters")
        plt.xlabel("HDBSCAN Cluster ID")
        plt.ylabel("Vocal Density")
        box_path = os.path.join(output_dir, 'vocal_density_boxplot.png')
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("Exporting Human-Readable Audio-Action Explanations...")
    explain_path = os.path.join(output_dir, 'audio_cluster_explanations.md')
    try:
        with open(explain_path, 'w') as f:
            f.write("# Audio-Action Contextual Explanations\n\n")
            f.write("This document maps the top 20 StepMania geometric patterns to their underlying audio triggers based on sliding Multi-Measure Chops.\n\n")
            
            for cid in top_clusters:
                audio_prof = cluster_means.loc[cid]
                
                # Fetch typical physical features for this cluster to cross-reference
                phys_query = f"SELECT avg(density) as d, avg(jumps) as j, avg(longest_diff_chain) as chain FROM stepmania_features s JOIN measure_cluster_assignments m ON s.run_id=m.run_id AND s.measure_idx=m.measure_idx AND s.chop_length=m.chop_length WHERE m.cluster_id={cid} AND m.run_id='{run_id}'"
                
                try:
                    phys_df = pd.read_sql(phys_query, conn)
                    if not phys_df.empty:
                        d, j, chain = phys_df.iloc[0]
                        shape_desc = f"{int(chain)}-step continuous staircases/trills" if chain and chain > 3 else "sporadic rhythmic steps"
                        jumps_desc = f"heavy jumping patterns ({j:.1f} jumps/chop)" if j and j > 4 else "mostly single notes"
                        
                        audio_triggers = []
                        if audio_prof.get('onset_density', 0) > df['onset_density'].mean() * 1.5:
                            audio_triggers.append("rapid Drum/Percussive strikes (High Onset Density)")
                        if audio_prof.get('spectral_centroid', 0) > df['spectral_centroid'].mean() * 1.2:
                            audio_triggers.append("high-frequency energy spikes (e.g., Cymbals/Synths)")
                        if audio_prof.get('rms_energy', 0) > df['rms_energy'].mean() * 1.3:
                            audio_triggers.append("very loud musical intensity (High RMS)")
                        if audio_prof.get('vocal_density', 0) > df['vocal_density'].mean() * 1.5:
                            audio_triggers.append("intense vocal sequences")
                            
                        trigger_text = ", ".join(audio_triggers) if audio_triggers else "steady background rhythms"
                        
                        f.write(f"### Cluster {cid}\n")
                        f.write(f"- **Physical Geography:** Predominantly features {shape_desc} mixed with {jumps_desc}.\n")
                        f.write(f"- **Audio Correlation:** These geometric motifs are heavily correlated with **{trigger_text}** in the audio track.\n\n")
                except Exception as ex:
                    pass
    except Exception as e:
        print(f"Failed to write explanations: {e}")

    print("Correlations successfully visualised and explained!")

def main():
    parser = argparse.ArgumentParser(description="Process beatmap datasets to discover patterns.")
    parser.add_argument('--target_dir', type=str, default='src/musicForBeatmap/',
                        help="Path to the master directory containing .ssc/.sm files")
    parser.add_argument('--output_dir', type=str, default='pattern_finding_results/',
                        help="Output directory for the UMAP PNG plot")
    parser.add_argument('--db_path', type=str, default='pattern_finding_approach/processed_files.db',
                        help="Path to the SQLite database (stores ALL results)")
    parser.add_argument('--encoding', type=str, default='features',
                        choices=['ohe', 'features', 'relative', 'rle', 'features+audio'],
                        help=(
                            "Encoding strategy for the topology pipeline:\n"
                            "  ohe      – One-Hot Encoding of canonical 192x4 matrix (most detailed)\n"
                            "  features – physical sequence traits (shift-invariant)\n"
                            "  relative – Column-delta histograms (shift-invariant)\n"
                            "  rle      – Run-length compression stats (stretch-invariant)\n"
                            "  features+audio – Semantic motif contextual fusion"
                        ))
    parser.add_argument('--chop_mode', type=str, default='both',
                        choices=['sliding', 'discrete', 'both'],
                        help="Data chunking logic: overlapping (sliding) or algorithmic continuous sweeps (discrete).")
    parser.add_argument('--export_csv', action='store_true',
                        help="Also export all DB result tables as CSV files (optional)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Open DB once — passed everywhere so all results land in the same file
    conn = init_database(args.db_path)

    # Unique ID for this run so multiple runs don't overwrite each other
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nRun ID: {run_id}  |  DB: {args.db_path}\n")

    raw_measures, seq_measures, measure_info, diff_stats, chars = process_dataset(args.target_dir, args.db_path, run_id, args.chop_mode)

    now = datetime.now().isoformat()
    if not measure_info:
        return

    # ------------------------------------------------------------------ #
    # 1. Difficulty 4-column breakdown → difficulty_breakdown table
    # ------------------------------------------------------------------ #
    breakdown_rows = []
    for diff, stats in sorted(diff_stats.items()):
        skipped = stats['total_charts'] - stats['files_with_4col']
        breakdown_rows.append((
            run_id, diff,
            stats['total_charts'], stats['files_with_4col'],
            stats['total_4col_measures'], skipped, now
        ))
    conn.executemany(
        "INSERT INTO difficulty_breakdown "
        "(run_id, difficulty, total_charts_in_files, charts_with_4col, "
        "total_4col_measures, skipped_charts_non4col, saved_at) "
        "VALUES (?,?,?,?,?,?,?)",
        breakdown_rows
    )
    conn.commit()
    print(f"\nDifficulty breakdown saved to DB (table: difficulty_breakdown, run_id={run_id})")
    # Print a preview to terminal
    df_bd = pd.read_sql(
        "SELECT difficulty, total_charts_in_files, charts_with_4col, "
        "total_4col_measures, skipped_charts_non4col "
        "FROM difficulty_breakdown WHERE run_id=? "
        "ORDER BY total_4col_measures DESC",
        conn, params=(run_id,)
    )
    print(df_bd.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2. Character distributions → character_distributions table
    # ------------------------------------------------------------------ #
    standard_chars = {'0', '1', '2', '3', '4', 'M', 'L', 'F', 'K', 'H'}
    char_rows = [
        (run_id, ch, diff, cnt, "No" if ch in standard_chars else "Yes", now)
        for (ch, diff), cnt in chars.items()
    ]
    conn.executemany(
        "INSERT INTO character_distributions "
        "(run_id, character, difficulty, count, is_new_pattern, saved_at) VALUES (?,?,?,?,?,?)",
        char_rows
    )
    conn.commit()
    print(f"\nCharacter distributions saved to DB (table: character_distributions, run_id={run_id})")
    df_ch = pd.read_sql(
        "SELECT character, difficulty, count, is_new_pattern FROM character_distributions "
        "WHERE run_id=? ORDER BY count DESC LIMIT 15",
        conn, params=(run_id,)
    )
    print(df_ch.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 3 & 4. ML Pipelines — topology + sequential (write directly to DB)
    # ------------------------------------------------------------------ #
    if measure_info:
        try:
            run_topology_pipeline(raw_measures, measure_info, conn, run_id, args.output_dir, args.encoding)
        except Exception as e:
            print(f"Topology Pipeline failed: {e}")

        try:
            run_sequential_pipeline(seq_measures, conn, run_id, args.output_dir)
        except Exception as e:
            print(f"Sequential Pipeline failed: {e}")
            
        try:
            generate_audio_cluster_correlations(conn, run_id, args.output_dir)
        except Exception as e:
            print(f"Audio-Cluster Correlation extraction failed: {e}")

    # ------------------------------------------------------------------ #
    # Optional CSV export  (--export_csv flag)
    # ------------------------------------------------------------------ #
    if args.export_csv:
        print("\n-- Exporting DB tables to CSV --")
        tables = [
            ("difficulty_breakdown",    "difficulty_4col_breakdown.csv"),
            ("character_distributions", "character_distributions.csv"),
            ("cluster_counts",          "hdbscan_cluster_counts.csv"),
            ("measure_cluster_assignments", "measure_cluster_assignments.csv"),
            ("cluster_transitions",     "macro_cluster_transitions.csv"),
            ("markov_transitions",      "markov_chain_transitions.csv"),
            ("prefixspan_patterns",     "prefixspan_frequent_patterns.csv"),
            ("audio_features",          "audio_features_metrics.csv"),
        ]
        for table, fname in tables:
            df = pd.read_sql(f"SELECT * FROM {table} WHERE run_id=?", conn, params=(run_id,))
            out = os.path.join(args.output_dir, fname)
            df.to_csv(out, index=False)
            print(f"  Exported {table} → {out}")

    conn.close()
    print(f"\nAll results stored in: {args.db_path}")
    print("Done!")

if __name__ == "__main__":
    main()
