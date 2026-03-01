"""
LLM Beatmap Metrics  (Issue #7)

Comprehensive evaluation metrics for LLM-generated DDR beatmaps.

Metrics implemented:
1. Structural Validity   – malformed rows, invalid chars, orphaned holds (LLM hallucinations)
2. Beat Alignment        – % of steps landing on detected beats        (wraps beatmap_validator)
3. Onset Alignment       – % of steps landing on musical onsets        (wraps beatmap_validator)
4. DDR Pattern Analysis  – streams, jacks, crossovers, jumps/hands/quads
5. Pattern Diversity     – unique step combos out of 256 possible (4 states × 4 columns)
6. Difficulty Distribution – NPS-based difficulty profile vs easy/med/hard thresholds
7. Step Count Statistics – total steps, arrows, NPS, vs original if provided

Usage:
    from src.analysis.llm_beatmap_metrics import compute_all_metrics, print_report

    result = compute_all_metrics(
        beatmap_path="generated.txt",
        audio_path="song.ogg",   # optional – needed for beat/onset alignment
        bpm=180.0,               # optional – needed for timing metrics
        offset=-0.028,           # optional
        original_path="orig.txt" # optional – enables step-count comparison
    )
    print_report(result)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StructuralValidity:
    """Metric 1 – structural integrity of the beatmap text."""
    valid: bool
    total_rows: int                 # rows parsed (excluding commas / blanks)
    invalid_rows: int               # rows that don't match the 4-char note format
    invalid_row_examples: List[str] # first 5 bad rows for inspection
    orphaned_hold_tails: int        # '3' in a column with no preceding '2'
    unclosed_hold_heads: int        # '2' in a column that never gets a '3'
    empty_beatmap: bool             # file has no notes at all
    structural_score: float         # 0–100, higher is better


@dataclass
class PatternAnalysis:
    """Metric 4 – DDR-specific pattern classification."""
    total_active_rows: int    # rows with at least one note
    singles: int              # 1 simultaneous note
    jumps: int                # 2 simultaneous notes
    hands: int                # 3 simultaneous notes
    quads: int                # 4 simultaneous notes
    mines: int                # rows containing mine(s)
    holds: int                # hold-head rows (2 or 4)

    # Sequential pattern counts
    streams: int              # 4+ consecutive single-tap rows (no rests)
    jacks: int                # 3+ consecutive taps on the SAME column
    crossovers: int           # alternate-column runs ≥4 (L→R→L or U→D→U …)

    pattern_counts: Dict[str, int] = field(default_factory=dict)  # per-type breakdown


@dataclass
class PatternDiversity:
    """Metric 5 – how many unique step combinations appear."""
    # In a 4-column tap-only view: 2^4 = 16 possible tap patterns
    unique_tap_patterns: int
    possible_tap_patterns: int           # = 16
    tap_diversity_pct: float             # unique_tap / 16 * 100

    # Full 4-state (0,1,2,3,4 mapped to binary presence) × 4 columns → 256 combinations
    unique_full_patterns: int
    possible_full_patterns: int          # = 256
    full_diversity_pct: float

    most_common_patterns: List[Tuple[str, int]]   # top-5 (pattern, count)


@dataclass
class DifficultyDistribution:
    """Metric 6 – inferred difficulty from NPS profile."""
    mean_nps: float
    peak_nps: float                       # 95th percentile in 1-s windows
    inferred_difficulty: str              # Easy / Medium / Hard / Expert
    nps_by_second: List[float]            # NPS in each 1-s window
    sparse_sections_pct: float            # % of 1-s windows with 0 notes
    dense_sections_pct: float             # % of 1-s windows with ≥8 NPS


@dataclass
class StepCountStats:
    """Metric 7 – step counts and comparison with original."""
    total_step_rows: int      # rows with ≥1 note
    total_arrows: int         # individual arrow presses
    total_measures: int
    mean_arrows_per_measure: float

    # Comparison with reference (optional)
    original_step_rows: Optional[int] = None
    original_arrows: Optional[int] = None
    step_count_ratio: Optional[float] = None   # generated / original


@dataclass
class AlignmentMetrics:
    """Metric 2 & 3 – beat / onset alignment (requires audio)."""
    beat_alignment_pct: float
    onset_alignment_pct: float
    percussive_alignment_pct: float
    mean_onset_distance_ms: float
    mean_beat_distance_ms: float
    match_perfect: int
    match_beat: int
    match_onset: int
    match_perc: int
    match_bad: int
    match_out_of_bounds: int
    audio_available: bool = True


@dataclass
class LLMBeatmapMetrics:
    """Top-level container returned by compute_all_metrics()."""
    beatmap_file: str
    audio_file: Optional[str]
    bpm: Optional[float]
    offset: Optional[float]
    original_file: Optional[str]
    timestamp: str

    structural: StructuralValidity
    patterns: PatternAnalysis
    diversity: PatternDiversity
    difficulty: DifficultyDistribution
    step_count: StepCountStats
    alignment: Optional[AlignmentMetrics]   # None when no audio supplied


# ---------------------------------------------------------------------------
# Low-level parsers (self-contained, no dependency on other src modules)
# ---------------------------------------------------------------------------

def _load_measures(filepath: str) -> List[List[str]]:
    """Parse a .txt/.text beatmap file into a list of measures."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot open beatmap: {filepath}") from exc

    measures: List[List[str]] = []
    current: List[str] = []

    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line == ",":
            if current:
                measures.append(current)
                current = []
        else:
            current.append(line)

    if current:
        measures.append(current)

    return measures


def _is_valid_row(row: str) -> bool:
    return len(row) == 4 and all(c in "01234M" for c in row)


def _count_arrows(row: str) -> int:
    """Number of note-head characters in a row (1, 2, 4 = tap / hold / roll head)."""
    return sum(1 for c in row if c in "124")


# ---------------------------------------------------------------------------
# Metric 1 – Structural Validity
# ---------------------------------------------------------------------------

def _check_structural_validity(measures: List[List[str]]) -> StructuralValidity:
    total_rows = 0
    invalid_rows = 0
    invalid_examples: List[str] = []
    orphaned_tails = 0
    unclosed_heads = 0
    has_any_note = False

    for measure in measures:
        hold_open = [False, False, False, False]   # per-column open hold

        for row in measure:
            total_rows += 1

            if not _is_valid_row(row):
                invalid_rows += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(repr(row))
                continue

            for col in range(4):
                ch = row[col]
                if ch in "14":                 # tap or roll head – fine
                    has_any_note = True
                elif ch == "2":                # hold head
                    hold_open[col] = True
                    has_any_note = True
                elif ch == "3":                # hold tail
                    if not hold_open[col]:
                        orphaned_tails += 1
                    else:
                        hold_open[col] = False

        # any hold that never closed within this measure counts as unclosed
        for col in range(4):
            if hold_open[col]:
                unclosed_heads += 1
                hold_open[col] = False         # reset for next measure

    valid_rows = total_rows - invalid_rows
    structural_score = (valid_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    return StructuralValidity(
        valid=invalid_rows == 0 and orphaned_tails == 0 and unclosed_heads == 0,
        total_rows=total_rows,
        invalid_rows=invalid_rows,
        invalid_row_examples=invalid_examples,
        orphaned_hold_tails=orphaned_tails,
        unclosed_hold_heads=unclosed_heads,
        empty_beatmap=not has_any_note,
        structural_score=structural_score,
    )


# ---------------------------------------------------------------------------
# Metric 4 – DDR Pattern Analysis
# ---------------------------------------------------------------------------

def _classify_patterns(measures: List[List[str]]) -> PatternAnalysis:
    singles = jumps = hands = quads = mines = holds = 0

    # For sequential pattern detection we need a flat list of (arrows per column)
    tap_sequence: List[Tuple[int, ...]] = []   # (L, D, U, R) 1/0 per active row

    for measure in measures:
        for row in measure:
            if not _is_valid_row(row):
                continue

            note_count = _count_arrows(row)
            mine_count = sum(1 for c in row if c == "M")
            hold_count = sum(1 for c in row if c == "2")

            if mine_count:
                mines += 1
            if hold_count:
                holds += hold_count

            if note_count == 1:
                singles += 1
            elif note_count == 2:
                jumps += 1
            elif note_count == 3:
                hands += 1
            elif note_count == 4:
                quads += 1

            if note_count >= 1:
                tap_sequence.append(
                    tuple(1 if row[i] in "124" else 0 for i in range(4))
                )

    # ------------------------------------------------------------------
    # Sequential patterns on the flat tap_sequence
    # ------------------------------------------------------------------
    streams = _count_streams(tap_sequence)
    jacks = _count_jacks(tap_sequence)
    crossovers = _count_crossovers(tap_sequence)

    total_active = singles + jumps + hands + quads

    pattern_counts = {
        "single": singles,
        "jump": jumps,
        "hand": hands,
        "quad": quads,
        "mine": mines,
        "hold_head": holds,
    }

    return PatternAnalysis(
        total_active_rows=total_active,
        singles=singles,
        jumps=jumps,
        hands=hands,
        quads=quads,
        mines=mines,
        holds=holds,
        streams=streams,
        jacks=jacks,
        crossovers=crossovers,
        pattern_counts=pattern_counts,
    )


def _count_streams(seq: List[Tuple[int, ...]], min_len: int = 4) -> int:
    """Count non-overlapping runs of ≥min_len consecutive *single* active rows."""
    count = run = 0
    for tup in seq:
        if sum(tup) == 1:   # exactly one arrow
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count


def _count_jacks(seq: List[Tuple[int, ...]], min_len: int = 3) -> int:
    """Count non-overlapping runs of ≥min_len taps on the same single column."""
    count = 0
    i = 0
    while i < len(seq):
        tup = seq[i]
        if sum(tup) == 1:          # single tap row
            col = tup.index(1)
            run = 1
            j = i + 1
            while j < len(seq) and seq[j] == tup:
                run += 1
                j += 1
            if run >= min_len:
                count += 1
            i = j
        else:
            i += 1
    return count


def _count_crossovers(seq: List[Tuple[int, ...]], min_len: int = 4) -> int:
    """Count runs of alternating left↔right or up↔down single taps."""
    LR = ({0}, {3})   # columns 0 (Left) and 3 (Right)
    UD = ({1}, {2})   # columns 1 (Down) and 2 (Up)
    count = run = 0
    last_col: Optional[int] = None
    last_pair: Optional[Tuple] = None

    for tup in seq:
        if sum(tup) != 1:
            if run >= min_len:
                count += 1
            run = 0
            last_col = last_pair = None
            continue

        col = tup.index(1)

        if last_col is None:
            run = 1
            last_col = col
            # Determine pair
            if col in LR[0] or col in LR[1]:
                last_pair = LR
            elif col in UD[0] or col in UD[1]:
                last_pair = UD
            else:
                last_pair = None
        else:
            # Continue if same pair and alternates
            if last_pair and col != last_col and (
                (col in last_pair[0] and last_col in last_pair[1]) or
                (col in last_pair[1] and last_col in last_pair[0])
            ):
                run += 1
                last_col = col
            else:
                if run >= min_len:
                    count += 1
                run = 1
                last_col = col
                if col in LR[0] or col in LR[1]:
                    last_pair = LR
                elif col in UD[0] or col in UD[1]:
                    last_pair = UD
                else:
                    last_pair = None

    if run >= min_len:
        count += 1
    return count


# ---------------------------------------------------------------------------
# Metric 5 – Pattern Diversity
# ---------------------------------------------------------------------------

def _compute_diversity(measures: List[List[str]]) -> PatternDiversity:
    tap_patterns: Counter = Counter()
    full_patterns: Counter = Counter()

    for measure in measures:
        for row in measure:
            if not _is_valid_row(row):
                continue
            if row == "0000":
                continue

            # Tap-only view: 1 if any note present in that column, else 0
            tap_key = "".join("1" if row[i] in "124" else "0" for i in range(4))
            tap_patterns[tap_key] += 1

            # Full 4-state (0,1,2,3,4 → kept as-is, M mapped to 5)
            full_key = row.replace("M", "5")
            full_patterns[full_key] += 1

    unique_tap = len(tap_patterns)
    unique_full = len(full_patterns)

    most_common = [(k, v) for k, v in tap_patterns.most_common(5)]

    return PatternDiversity(
        unique_tap_patterns=unique_tap,
        possible_tap_patterns=16,
        tap_diversity_pct=unique_tap / 16 * 100.0,
        unique_full_patterns=unique_full,
        possible_full_patterns=256,
        full_diversity_pct=unique_full / 256 * 100.0,
        most_common_patterns=most_common,
    )


# ---------------------------------------------------------------------------
# Metric 6 – Difficulty Distribution
# ---------------------------------------------------------------------------

def _compute_difficulty(
    measures: List[List[str]],
    bpm: Optional[float],
    offset: Optional[float],
) -> DifficultyDistribution:
    """Compute NPS-based difficulty profile.

    If bpm/offset are unknown we fall back to measure index to approximate time.
    """
    beats_per_measure = 4
    seconds_per_beat = 60.0 / bpm if bpm else 0.5  # fallback: 120 BPM
    start_time = offset if offset is not None else 0.0

    note_times: List[float] = []

    for m_idx, measure in enumerate(measures):
        lines = len(measure)
        if lines == 0:
            continue
        beats_per_line = beats_per_measure / lines
        m_start_beat = m_idx * beats_per_measure

        for l_idx, row in enumerate(measure):
            if not _is_valid_row(row):
                continue
            if _count_arrows(row) == 0:
                continue
            beat_pos = m_start_beat + l_idx * beats_per_line
            note_times.append(start_time + beat_pos * seconds_per_beat)

    if not note_times:
        return DifficultyDistribution(
            mean_nps=0.0,
            peak_nps=0.0,
            inferred_difficulty="Unknown",
            nps_by_second=[],
            sparse_sections_pct=100.0,
            dense_sections_pct=0.0,
        )

    duration = int(note_times[-1]) + 1
    nps_by_second = [0.0] * duration

    for t in note_times:
        sec = int(t)
        if 0 <= sec < duration:
            nps_by_second[sec] += 1

    mean_nps = float(np.mean(nps_by_second)) if nps_by_second else 0.0
    peak_nps = float(np.percentile(nps_by_second, 95)) if nps_by_second else 0.0

    sparse = sum(1 for n in nps_by_second if n == 0) / max(duration, 1) * 100.0
    dense = sum(1 for n in nps_by_second if n >= 8) / max(duration, 1) * 100.0

    if peak_nps < 4:
        difficulty = "Easy"
    elif peak_nps < 8:
        difficulty = "Medium"
    elif peak_nps < 12:
        difficulty = "Hard"
    else:
        difficulty = "Expert"

    return DifficultyDistribution(
        mean_nps=round(mean_nps, 2),
        peak_nps=round(peak_nps, 2),
        inferred_difficulty=difficulty,
        nps_by_second=[round(n, 2) for n in nps_by_second],
        sparse_sections_pct=round(sparse, 1),
        dense_sections_pct=round(dense, 1),
    )


# ---------------------------------------------------------------------------
# Metric 7 – Step Count
# ---------------------------------------------------------------------------

def _compute_step_count(
    measures: List[List[str]],
    original_measures: Optional[List[List[str]]] = None,
) -> StepCountStats:
    def _tally(mlist: List[List[str]]) -> Tuple[int, int]:
        steps = arrows = 0
        for measure in mlist:
            for row in measure:
                if not _is_valid_row(row):
                    continue
                n = _count_arrows(row)
                if n > 0:
                    steps += 1
                    arrows += n
        return steps, arrows

    gen_steps, gen_arrows = _tally(measures)
    total_measures = len(measures)
    mean_apm = gen_arrows / total_measures if total_measures > 0 else 0.0

    orig_steps = orig_arrows = ratio = None
    if original_measures is not None:
        orig_steps, orig_arrows = _tally(original_measures)
        ratio = gen_steps / orig_steps if orig_steps > 0 else None

    return StepCountStats(
        total_step_rows=gen_steps,
        total_arrows=gen_arrows,
        total_measures=total_measures,
        mean_arrows_per_measure=round(mean_apm, 2),
        original_step_rows=orig_steps,
        original_arrows=orig_arrows,
        step_count_ratio=round(ratio, 3) if ratio is not None else None,
    )


# ---------------------------------------------------------------------------
# Metrics 2 & 3 – Beat / Onset Alignment (wraps beatmap_validator)
# ---------------------------------------------------------------------------

def _compute_alignment(
    audio_path: str,
    beatmap_path: str,
    bpm: float,
    offset: float,
) -> AlignmentMetrics:
    """Delegate to beatmap_validator.validate_beatmap() and repack results."""
    try:
        # Lazy import – beatmap_validator needs librosa which may not be available
        # when running metrics without audio
        sys.path.insert(0, os.path.dirname(__file__))
        from beatmap_validator import validate_beatmap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "beatmap_validator could not be imported. "
            "Ensure librosa is installed to use audio-based metrics."
        ) from exc

    vr = validate_beatmap(audio_path, beatmap_path, bpm, offset)

    return AlignmentMetrics(
        beat_alignment_pct=round(vr.beat_alignment_percentage, 2),
        onset_alignment_pct=round(vr.onset_alignment_percentage, 2),
        percussive_alignment_pct=round(vr.percussive_alignment_percentage, 2),
        mean_onset_distance_ms=round(vr.mean_onset_distance_ms, 2),
        mean_beat_distance_ms=round(vr.mean_beat_distance_ms, 2),
        match_perfect=vr.match_perfect,
        match_beat=vr.match_beat,
        match_onset=vr.match_onset,
        match_perc=vr.match_perc,
        match_bad=vr.match_bad,
        match_out_of_bounds=vr.match_out_of_bounds,
        audio_available=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_metrics(
    beatmap_path: str,
    audio_path: Optional[str] = None,
    bpm: Optional[float] = None,
    offset: Optional[float] = None,
    original_path: Optional[str] = None,
) -> LLMBeatmapMetrics:
    """Compute all evaluation metrics for a generated beatmap.

    Args:
        beatmap_path:   Path to generated .txt / .text beatmap file.
        audio_path:     Path to audio file (.ogg / .mp3 / .wav).  Optional.
                        Required for beat / onset alignment metrics.
        bpm:            Song BPM.  Optional – used for timing calculations.
        offset:         Song start offset in seconds.  Optional.
        original_path:  Path to ground-truth beatmap for step-count comparison.
                        Optional.

    Returns:
        LLMBeatmapMetrics dataclass with all computed metrics.
    """
    measures = _load_measures(beatmap_path)
    original_measures = _load_measures(original_path) if original_path else None

    structural = _check_structural_validity(measures)
    patterns = _classify_patterns(measures)
    diversity = _compute_diversity(measures)
    difficulty = _compute_difficulty(measures, bpm, offset)
    step_count = _compute_step_count(measures, original_measures)

    alignment: Optional[AlignmentMetrics] = None
    if audio_path and bpm is not None and offset is not None:
        try:
            alignment = _compute_alignment(audio_path, beatmap_path, bpm, offset)
        except Exception as exc:
            print(f"[llm_beatmap_metrics] Alignment skipped: {exc}")

    return LLMBeatmapMetrics(
        beatmap_file=beatmap_path,
        audio_file=audio_path,
        bpm=bpm,
        offset=offset,
        original_file=original_path,
        timestamp=datetime.now().isoformat(),
        structural=structural,
        patterns=patterns,
        diversity=diversity,
        difficulty=difficulty,
        step_count=step_count,
        alignment=alignment,
    )


def save_metrics_json(metrics: LLMBeatmapMetrics, output_path: str) -> None:
    """Serialise metrics to a JSON file."""
    data = asdict(metrics)
    # most_common_patterns contains tuples → convert to lists for JSON
    if "diversity" in data and "most_common_patterns" in data["diversity"]:
        data["diversity"]["most_common_patterns"] = [
            list(p) for p in data["diversity"]["most_common_patterns"]
        ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def print_report(metrics: LLMBeatmapMetrics) -> None:
    """Print a human-readable summary to stdout."""
    SEP = "=" * 62

    print(f"\n{SEP}")
    print("  LLM BEATMAP EVALUATION REPORT")
    print(SEP)
    print(f"  File    : {os.path.basename(metrics.beatmap_file)}")
    if metrics.original_file:
        print(f"  Original: {os.path.basename(metrics.original_file)}")
    if metrics.bpm:
        print(f"  BPM     : {metrics.bpm}  Offset: {metrics.offset}s")
    print(f"  Time    : {metrics.timestamp}")

    # 1 – Structural Validity
    s = metrics.structural
    status = "PASS" if s.valid else "FAIL"
    print(f"\n[1] Structural Validity  [{status}]")
    print(f"    Total rows          : {s.total_rows}")
    print(f"    Invalid rows        : {s.invalid_rows}")
    print(f"    Orphaned hold tails : {s.orphaned_hold_tails}")
    print(f"    Unclosed hold heads : {s.unclosed_hold_heads}")
    print(f"    Empty beatmap       : {s.empty_beatmap}")
    print(f"    Structural score    : {s.structural_score:.1f}%")
    if s.invalid_row_examples:
        print(f"    Bad row examples    : {', '.join(s.invalid_row_examples)}")

    # 2 & 3 – Alignment
    if metrics.alignment:
        a = metrics.alignment
        print(f"\n[2] Beat Alignment")
        print(f"    On beat             : {a.beat_alignment_pct:.1f}%")
        print(f"    Mean dist to beat   : {a.mean_beat_distance_ms:.1f} ms")
        print(f"\n[3] Onset Alignment")
        print(f"    On onset            : {a.onset_alignment_pct:.1f}%")
        print(f"    On percussive       : {a.percussive_alignment_pct:.1f}%")
        print(f"    Mean dist to onset  : {a.mean_onset_distance_ms:.1f} ms")
        print(f"    Perfect (beat+onset): {a.match_perfect}")
        print(f"    Beat only           : {a.match_beat}")
        print(f"    Onset only          : {a.match_onset}")
        print(f"    Percussive only     : {a.match_perc}")
        print(f"    Unaligned           : {a.match_bad}")
        if a.match_out_of_bounds:
            print(f"    Out of bounds       : {a.match_out_of_bounds}")
    else:
        print(f"\n[2/3] Beat / Onset Alignment  [SKIPPED – no audio provided]")

    # 4 – Pattern Analysis
    p = metrics.patterns
    total = max(p.total_active_rows, 1)
    print(f"\n[4] DDR Pattern Analysis  (active rows: {p.total_active_rows})")
    print(f"    Singles   : {p.singles:5d}  ({p.singles/total*100:5.1f}%)")
    print(f"    Jumps     : {p.jumps:5d}  ({p.jumps/total*100:5.1f}%)")
    print(f"    Hands     : {p.hands:5d}  ({p.hands/total*100:5.1f}%)")
    print(f"    Quads     : {p.quads:5d}  ({p.quads/total*100:5.1f}%)")
    print(f"    Holds     : {p.holds:5d}")
    print(f"    Mines     : {p.mines:5d}")
    print(f"    Streams   : {p.streams:5d}  (runs of 4+ single taps)")
    print(f"    Jacks     : {p.jacks:5d}  (3+ same-column repeats)")
    print(f"    Crossovers: {p.crossovers:5d}  (4+ alternating L↔R or U↔D)")

    # 5 – Pattern Diversity
    d = metrics.diversity
    print(f"\n[5] Pattern Diversity")
    print(f"    Unique tap patterns  : {d.unique_tap_patterns} / {d.possible_tap_patterns}"
          f"  ({d.tap_diversity_pct:.1f}%)")
    print(f"    Unique full patterns : {d.unique_full_patterns} / {d.possible_full_patterns}"
          f"  ({d.full_diversity_pct:.1f}%)")
    if d.most_common_patterns:
        top = ", ".join(f"{pat}×{cnt}" for pat, cnt in d.most_common_patterns[:3])
        print(f"    Top-3 tap patterns  : {top}")

    # 6 – Difficulty Distribution
    df = metrics.difficulty
    print(f"\n[6] Difficulty Distribution")
    print(f"    Inferred difficulty  : {df.inferred_difficulty}")
    print(f"    Mean NPS             : {df.mean_nps:.2f}")
    print(f"    Peak NPS (p95)       : {df.peak_nps:.2f}")
    print(f"    Sparse sections (0NPS): {df.sparse_sections_pct:.1f}%")
    print(f"    Dense  sections (≥8) : {df.dense_sections_pct:.1f}%")

    # 7 – Step Count
    sc = metrics.step_count
    print(f"\n[7] Step Count Statistics")
    print(f"    Total step rows      : {sc.total_step_rows}")
    print(f"    Total arrows         : {sc.total_arrows}")
    print(f"    Total measures       : {sc.total_measures}")
    print(f"    Mean arrows/measure  : {sc.mean_arrows_per_measure:.1f}")
    if sc.original_step_rows is not None:
        print(f"    Original step rows   : {sc.original_step_rows}")
        print(f"    Original arrows      : {sc.original_arrows}")
        print(f"    Step count ratio     : {sc.step_count_ratio:.3f}  (gen/orig)")

    print(f"\n{SEP}\n")
