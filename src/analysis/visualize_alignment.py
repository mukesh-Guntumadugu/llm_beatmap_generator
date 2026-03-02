"""
visualize_alignment.py — plain-text alignment report (no graphs)

Reads one or more generated beatmap CSVs, detects musical onsets/beats
from the audio, and prints a clean percentage summary for each beatmap.

Usage:
    python3 src/analysis/visualize_alignment.py \
        --audio  "path/to/song.ogg" \
        --beatmaps "path/to/beatmap.csv" \
        --output  "report.txt"   # optional, saves the same text to a file
"""

import argparse
import os
import sys

import librosa
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_step_times_from_csv(csv_path: str, tolerance_s: float = 0.001) -> np.ndarray:
    """Read time_ms column from a beatmap CSV, return seconds for active rows."""
    times = []
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            t_idx = header.index("time_ms")
            n_idx = header.index("notes")
        except ValueError:
            print(f"  [error] CSV missing time_ms or notes column: {csv_path}")
            return np.array([])

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(t_idx, n_idx):
                continue
            note = parts[n_idx].strip()
            if note in ("", "0000", ","):
                continue
            try:
                times.append(float(parts[t_idx]) / 1000.0)
            except ValueError:
                continue

    return np.array(sorted(times))


def nearest_distance(step_times: np.ndarray, ref_times: np.ndarray) -> np.ndarray:
    """For each step time, find the distance to the nearest reference time."""
    if len(ref_times) == 0:
        return np.full(len(step_times), np.inf)
    distances = []
    for t in step_times:
        idx = np.searchsorted(ref_times, t)
        d = np.inf
        if idx > 0:
            d = min(d, abs(t - ref_times[idx - 1]))
        if idx < len(ref_times):
            d = min(d, abs(t - ref_times[idx]))
        distances.append(d)
    return np.array(distances)


def alignment_pct(distances: np.ndarray, tol_ms: float) -> float:
    if len(distances) == 0:
        return 0.0
    return (distances <= tol_ms / 1000.0).sum() / len(distances) * 100.0


# ── Main ─────────────────────────────────────────────────────────────────────

def analyse(audio_path: str, csv_paths: list[str], tol_ms: float = 50.0) -> str:
    lines = []

    def p(s=""):
        lines.append(s)

    p(f"Audio  : {os.path.basename(audio_path)}")
    p(f"Tolerance: {tol_ms:.0f} ms")
    p("=" * 56)

    # Load audio features once
    p("Detecting onsets and beats...")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times     = librosa.frames_to_time(beat_frames, sr=sr)

    y_harm, y_perc = librosa.effects.hpss(y)
    perc_frames    = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=True)
    perc_times     = librosa.frames_to_time(perc_frames, sr=sr)

    p(f"Onsets detected  : {len(onset_times)}")
    p(f"Beats detected   : {len(beat_times)}")
    p(f"Perc. onsets     : {len(perc_times)}")
    p(f"Audio duration   : {duration:.1f}s")
    p()

    for csv_path in csv_paths:
        p("-" * 56)
        p(f"Beatmap: {os.path.basename(csv_path)}")

        step_times = load_step_times_from_csv(csv_path)

        if len(step_times) == 0:
            p("  (no active steps found)")
            p()
            continue

        # Filter out-of-bounds steps
        valid = step_times[step_times <= duration]
        oob   = len(step_times) - len(valid)

        d_onset = nearest_distance(valid, onset_times)
        d_beat  = nearest_distance(valid, beat_times)
        d_perc  = nearest_distance(valid, perc_times)

        onset_pct = alignment_pct(d_onset, tol_ms)
        beat_pct  = alignment_pct(d_beat,  tol_ms)
        perc_pct  = alignment_pct(d_perc,  tol_ms)

        on_onset = int((d_onset <= tol_ms / 1000).sum())
        on_beat  = int((d_beat  <= tol_ms / 1000).sum())
        on_perc  = int((d_perc  <= tol_ms / 1000).sum())
        total    = len(valid)

        p(f"  Total steps (active rows) : {len(step_times)}")
        if oob:
            p(f"  Out of bounds (ignored)   : {oob}")
        p(f"  Valid steps               : {total}")
        p()
        p(f"  Onset alignment  : {on_onset}/{total}  →  {onset_pct:.1f}%")
        p(f"  Beat  alignment  : {on_beat}/{total}  →  {beat_pct:.1f}%")
        p(f"  Perc. alignment  : {on_perc}/{total}  →  {perc_pct:.1f}%")
        p(f"  Mean dist onset  : {np.mean(d_onset)*1000:.1f} ms")
        p(f"  Mean dist beat   : {np.mean(d_beat)*1000:.1f} ms")
        p()

    p("=" * 56)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plain-text alignment report for beatmap CSVs")
    parser.add_argument("--audio",    required=True, help="Path to audio file")
    parser.add_argument("--beatmaps", required=True, nargs="+", help="One or more beatmap CSV paths")
    parser.add_argument("--output",   default=None,  help="Optional path to save report (txt)")
    parser.add_argument("--tol",      type=float, default=50.0, help="Tolerance in ms (default 50)")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    report = analyse(args.audio, args.beatmaps, tol_ms=args.tol)
    print(report)

    if args.output:
        # Always save as .txt regardless of extension passed
        out_path = os.path.splitext(args.output)[0] + ".txt"
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
