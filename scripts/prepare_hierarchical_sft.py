#!/usr/bin/env python3
"""
prepare_hierarchical_sft.py
============================
Phase 2 of the Hierarchical Generation Architecture.

For every song in the SQLite database, this script:
  1. Groups all measures by file_path + difficulty (via audio_features & measure_cluster_assignments).
  2. Slides a 5-second window across the song's timeline.
  3. Finds all measures whose time range overlaps that 5s window in ORDER.
  4. Extracts the exact cluster token sequence for those measures.
  5. Slices the corresponding 5-second .wav chunk using librosa.
  6. Writes a JSONL training record: { audio_path, text_prompt, text_response }.

The resulting text_response for each 5s window is:
    "<|cluster_8|> <|cluster_12|> <|cluster_4|>"

This is what the LLM is trained to predict - pure cluster sequences, zero raw arrows.

Usage (local):
    python scripts/prepare_hierarchical_sft.py

Usage (HPC Slurm):
    Adjust HPC_BASE_AUDIO_PATH and DB_PATH for cluster filesystem paths.
"""

import os
import sys
import json
import csv
import sqlite3
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("ERROR: librosa and soundfile are required. Install via: pip install librosa soundfile")
    sys.exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH         = os.path.join(REPO_ROOT, "pattern_finding_approach", "processed_files.db")
OUTPUT_DIR      = os.path.join(REPO_ROOT, "hierarchical_sft_dataset")
AUDIO_OUT_DIR   = os.path.join(OUTPUT_DIR, "audio")

# The base path the HPC cluster will see for the audio chunks.
HPC_BASE_AUDIO_PATH = "/data/mg546924/llm_beatmap_generator/hierarchical_sft_dataset/audio"

# Tempo Change Analysis CSV (produced by onsetdetection/extract_tempo_changes.py)
TEMPO_CSV_PATH  = os.path.join(REPO_ROOT, "onsetdetection", "Tempo_Change_Analysis.csv")

CHUNK_SIZE_SEC   = 5.0      # 5-second audio windows
SAMPLE_RATE      = 16000    # Resample all audio to 16kHz for model input
MIN_CLUSTERS     = 2        # Skip windows with fewer than 2 clusters (likely silence/padding)
# ───────────────────────────────────────────────────────────────────────────────


# ─── Tempo Lookup (from extract_tempo_changes.py output) ───────────────────────

def load_tempo_lookup(tempo_csv_path: str) -> dict:
    """
    Loads Tempo_Change_Analysis.csv into a dict keyed by Song_Name (folder basename).

    Returns:
        {
          "Springtime": {
              "is_variable":   True,
              "gt_timeline":   [{"time_sec": 0.0, "bpm": 181.68, ...}, ...],
              "global_bpm":    181.68,
          },
          ...
        }
    Returns empty dict if CSV not found (degrades gracefully).
    """
    if not os.path.exists(tempo_csv_path):
        print(f"  ⚠  Tempo CSV not found: {tempo_csv_path}")
        print("     Run: sbatch onsetdetection/slurm_tempo_extraction.sh first.")
        print("     Continuing WITHOUT tempo conditioning...\n")
        return {}

    lookup = {}
    with open(tempo_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song_name = row["Song_Name"].strip()
            try:
                gt_timeline = json.loads(row.get("GT_BPM_Timeline", "[]"))
            except json.JSONDecodeError:
                gt_timeline = []

            is_variable = row.get("Is_Variable_Tempo", "False").strip().lower() == "true"

            # Global BPM = first entry in GT timeline (beat 0 BPM)
            global_bpm = gt_timeline[0]["bpm"] if gt_timeline else None

            lookup[song_name] = {
                "is_variable": is_variable,
                "gt_timeline": gt_timeline,
                "global_bpm":  global_bpm,
            }

    print(f"  Loaded tempo data for {len(lookup)} songs.")
    return lookup


def get_active_bpm(time_sec: float, gt_timeline: list) -> float | None:
    """
    Returns the BPM that is active at a given time in seconds,
    based on the ground-truth BPM timeline from the .sm/.ssc file.

    If gt_timeline is empty, returns None.
    """
    if not gt_timeline:
        return None

    active_bpm = gt_timeline[0]["bpm"]  # default: first zone
    for zone in gt_timeline:
        if zone["time_sec"] <= time_sec:
            active_bpm = zone["bpm"]
        else:
            break
    return active_bpm


def build_tempo_prompt_fragment(win_start: float, win_mid: float, tempo_info: dict | None) -> str:
    """
    Builds the tempo conditioning fragment to inject into the prompt.

    Examples:
      Constant tempo:  "The song runs at a constant 210 BPM."
      Variable tempo:  "The song has tempo changes. At this moment (t=5.0s) the BPM is 181.7."
      No data:         ""  (empty string — degrades gracefully)
    """
    if not tempo_info:
        return ""

    gt_timeline  = tempo_info["gt_timeline"]
    is_variable  = tempo_info["is_variable"]
    active_bpm   = get_active_bpm(win_mid, gt_timeline)

    if active_bpm is None:
        return ""

    if is_variable:
        return (
            f"This song has tempo changes. "
            f"At this audio segment (t={win_mid:.1f}s) the BPM is {active_bpm:.1f}. "
        )
    else:
        return f"The song runs at a constant {active_bpm:.1f} BPM. "



def get_all_songs(conn):
    """
    Returns a deduplicated list of (file_path, difficulty) combos
    that have BOTH audio_features AND measure_cluster_assignments data.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT af.file_path, af.difficulty
        FROM audio_features af
        JOIN measure_cluster_assignments mca
          ON af.file_path  = mca.file_path
         AND af.difficulty = mca.difficulty
         AND af.run_id     = mca.run_id
        ORDER BY af.file_path, af.difficulty
    """)
    return cursor.fetchall()


def get_measures_for_song(conn, file_path, difficulty):
    """
    Returns all measures for a (file_path, difficulty) in chronological order,
    with their start_time, end_time, and cluster_id.
    Filters out noise cluster (-1).
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT af.measure_idx, af.start_time, af.end_time, mca.cluster_id
        FROM audio_features af
        JOIN measure_cluster_assignments mca
          ON af.file_path  = mca.file_path
         AND af.difficulty = mca.difficulty
         AND af.measure_idx = mca.measure_idx
         AND af.run_id     = mca.run_id
        WHERE af.file_path  = ?
          AND af.difficulty = ?
          AND mca.cluster_id != -1
        ORDER BY af.measure_idx ASC
    """, (file_path, difficulty))
    return cursor.fetchall()


def find_audio_file(file_path):
    """
    Given a relative file_path from the DB (e.g. src/musicForBeatmap/PackA/Song/Song.ssc),
    finds the actual audio file (.ogg, .mp3, .wav) next to the .ssc/.sm file.
    Returns None if no audio found.
    """
    # Try absolute path from repo root first
    abs_ssc = os.path.join(REPO_ROOT, file_path)
    song_dir = os.path.dirname(abs_ssc)
    song_stem = os.path.splitext(os.path.basename(abs_ssc))[0]

    # Try matching audio files in the same directory
    for ext in ['.ogg', '.mp3', '.wav']:
        # try stem match
        candidate = os.path.join(song_dir, song_stem + ext)
        if os.path.exists(candidate):
            return candidate
        # try any audio file in directory
    for fname in os.listdir(song_dir) if os.path.isdir(song_dir) else []:
        if fname.lower().endswith(('.ogg', '.mp3', '.wav')):
            return os.path.join(song_dir, fname)

    return None


def build_local_file_index():
    """
    Builds a lookup dict: filename_stem -> first matching local audio path.
    Used to resolve DB paths that don't match the local directory layout
    (e.g. 'Fraxtil 4' in DB but 'Fraxtil' locally).
    """
    print("Building local audio file index...")
    music_dir = os.path.join(REPO_ROOT, "src", "musicForBeatmap")
    index = {}
    for root, _, files in os.walk(music_dir):
        for f in files:
            if f.lower().endswith(('.ogg', '.mp3', '.wav')):
                stem = os.path.splitext(f)[0]
                if stem not in index:  # first match wins
                    index[stem] = os.path.join(root, f)
    print(f"  Indexed {len(index)} audio files.")
    return index


def slice_and_save_audio(y, sr, start_sec, end_sec, out_path):
    """
    Slices y[start_sec:end_sec], resamples to SAMPLE_RATE, and saves as wav.
    Returns True on success.
    """
    start_sample = int(start_sec * sr)
    end_sample   = int(end_sec   * sr)
    chunk = y[start_sample:end_sample]

    if len(chunk) == 0:
        return False

    # Resample to target sr if the source sr differs
    if sr != SAMPLE_RATE:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Skip near-silence
    if librosa.feature.rms(y=chunk).mean() < 0.004:
        return False

    sf.write(out_path, chunk, SAMPLE_RATE)
    return True


def process_song(conn, file_path, difficulty, audio_index, jsonl_file, stats, tempo_lookup=None):
    """
    Processes a single (file_path, difficulty) pair:
    - Loads audio
    - Slides 5-second windows
    - For each window, finds the ordered cluster sequence from DB
    - Saves audio chunk + JSONL record
    """
    # ── Find audio ──────────────────────────────────────────────────────────
    song_stem = os.path.splitext(os.path.basename(file_path))[0]
    audio_path = find_audio_file(file_path)

    # Fallback to global index by stem
    if not audio_path:
        audio_path = audio_index.get(song_stem)

    if not audio_path:
        stats["no_audio"] += 1
        return

    # ── Load audio ──────────────────────────────────────────────────────────
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"  [ERROR] Could not load {audio_path}: {e}")
        stats["load_error"] += 1
        return

    duration = librosa.get_duration(y=y, sr=sr)

    # ── Get ordered measures ────────────────────────────────────────────────
    measures = get_measures_for_song(conn, file_path, difficulty)
    if not measures:
        stats["no_measures"] += 1
        return

    # Build timeline: list of (start_time, end_time, cluster_id)
    timeline = [(m[1], m[2], m[3]) for m in measures]

    # ── Slide 5-second windows ──────────────────────────────────────────────
    song_key = f"{song_stem}_{difficulty}"
    window_idx = 0

    for win_start in np.arange(0, duration - CHUNK_SIZE_SEC + 0.1, CHUNK_SIZE_SEC):
        win_end = win_start + CHUNK_SIZE_SEC

        # Clamp to actual audio duration
        actual_end = min(win_end, duration)
        if actual_end - win_start < (CHUNK_SIZE_SEC * 0.8):
            continue  # Skip partial trailing window

        # ── Find overlapping measures in order ──────────────────────────────
        # A measure overlaps the window if its interval [start, end] intersects [win_start, win_end]
        overlapping_clusters = []
        for (m_start, m_end, cluster_id) in timeline:
            # Overlap condition: measure starts before window ends AND measure ends after window starts
            if m_start < win_end and m_end > win_start:
                overlapping_clusters.append(cluster_id)

        if len(overlapping_clusters) < MIN_CLUSTERS:
            continue  # Not enough musical content in this window

        # ── Build the target cluster token sequence ─────────────────────────
        cluster_tokens = " ".join(f"<|cluster_{c}|>" for c in overlapping_clusters)

        # ── Save the audio chunk ────────────────────────────────────────────
        chunk_name = f"{song_key}_w{window_idx:04d}.wav"
        # Sanitize filename chars
        chunk_name = chunk_name.replace("/", "_").replace("'", "").replace(" ", "_")
        local_chunk_path = os.path.join(AUDIO_OUT_DIR, chunk_name)
        hpc_chunk_path   = f"{HPC_BASE_AUDIO_PATH}/{chunk_name}"

        if not slice_and_save_audio(y, sr, win_start, actual_end, local_chunk_path):
            stats["silence_skipped"] += 1
            continue

        # ── Tempo conditioning ──────────────────────────────────────────────
        win_mid = (win_start + actual_end) / 2.0
        tempo_info = tempo_lookup.get(song_stem) if tempo_lookup else None
        tempo_fragment = build_tempo_prompt_fragment(win_start, win_mid, tempo_info)
        active_bpm = get_active_bpm(win_mid, tempo_info["gt_timeline"]) if tempo_info else None
        is_variable = tempo_info["is_variable"] if tempo_info else False

        # ── Build prompt ────────────────────────────────────────────────────
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this {round(actual_end - win_start, 1)}s audio segment. "
            f"The difficulty is {difficulty}. "
            f"{tempo_fragment}"
            "Predict the ordered sequence of rhythmic pattern cluster tokens "
            "that best matches the audio's energy, density, and rhythm."
        )

        # ── Write JSONL record ──────────────────────────────────────────────
        record = {
            "id":               f"{song_key}_w{window_idx:04d}",
            "audio_path":       hpc_chunk_path,
            "local_audio_path": local_chunk_path,
            "file_path":        file_path,
            "difficulty":       difficulty,
            "window_start_sec": round(float(win_start), 3),
            "window_end_sec":   round(float(actual_end), 3),
            "num_clusters":     len(overlapping_clusters),
            # ── Tempo fields (new) ──────────────────────────────────────────
            "bpm_at_window":    round(active_bpm, 2) if active_bpm else None,
            "is_variable_tempo": is_variable,
            "tempo_fragment":   tempo_fragment,
            # ── Text I/O ────────────────────────────────────────────────────
            "text_prompt":      prompt,
            "text_response":    cluster_tokens,
        }
        jsonl_file.write(json.dumps(record) + "\n")

        stats["total_windows"] += 1
        window_idx += 1

    stats["songs_processed"] += 1


def main():
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    out_jsonl_path = os.path.join(OUTPUT_DIR, "hierarchical_train.jsonl")

    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH, timeout=30)

    audio_index  = build_local_file_index()
    tempo_lookup = load_tempo_lookup(TEMPO_CSV_PATH)   # ← load tempo data

    songs = get_all_songs(conn)
    print(f"Found {len(songs)} unique (file_path, difficulty) combos to process.")
    print(f"Tempo data available for {len(tempo_lookup)} songs.\n")

    stats = {
        "songs_processed":    0,
        "total_windows":      0,
        "no_audio":           0,
        "load_error":         0,
        "no_measures":        0,
        "silence_skipped":    0,
        "tempo_conditioned":  0,   # windows that got a real BPM injected
    }

    with open(out_jsonl_path, "w", encoding="utf-8") as f_out:
        for i, (file_path, difficulty) in enumerate(songs):
            song_name = os.path.basename(os.path.dirname(file_path))
            print(f"  [{i+1:>4}/{len(songs)}] {song_name} [{difficulty}]", end="  ")
            before = stats["total_windows"]
            process_song(conn, file_path, difficulty, audio_index, f_out, stats,
                         tempo_lookup=tempo_lookup)     # ← pass tempo data
            added = stats["total_windows"] - before
            print(f"→ +{added} windows")

    conn.close()

    print("\n" + "="*60)
    print("✅  HIERARCHICAL SFT DATASET COMPLETE")
    print("="*60)
    print(f"  Songs processed  : {stats['songs_processed']}")
    print(f"  Total windows    : {stats['total_windows']}")
    print(f"  No audio found   : {stats['no_audio']}")
    print(f"  Load errors      : {stats['load_error']}")
    print(f"  Silence skipped  : {stats['silence_skipped']}")
    print(f"\n  JSONL output     : {out_jsonl_path}")
    print(f"  Audio chunks dir : {AUDIO_OUT_DIR}")
    print("="*60)

    # Print a sample record
    print("\n── Sample Training Record ─────────────────────────────────")
    with open(out_jsonl_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            print(f"  id            : {sample['id']}")
            print(f"  audio_path    : {sample['audio_path']}")
            print(f"  difficulty    : {sample['difficulty']}")
            print(f"  window        : {sample['window_start_sec']}s → {sample['window_end_sec']}s")
            print(f"  num_clusters  : {sample['num_clusters']}")
            print(f"  text_prompt   : {sample['text_prompt'][:80]}...")
            print(f"  text_response : {sample['text_response']}")
            break


if __name__ == "__main__":
    main()
