#!/usr/bin/env python3
"""
Beatmap Alignment Report — plain-text percentage summary.

Usage:
  python3 src/analysis/visualize_alignment.py \
    --audio "path/to/song.ogg" \
    --beatmaps "path/to/gemini.csv" "path/to/qwen.csv" \
    --output "report.txt"   # optional
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import librosa
from src.analysis.beatmap_validator import detect_onsets, detect_beats, load_beatmap_measures, get_step_times


def _nearest(step_times, ref_times):
    """Distance from each step to its nearest reference time."""
    if len(ref_times) == 0:
        return np.full(len(step_times), np.inf)
    dists = []
    for t in step_times:
        idx = np.searchsorted(ref_times, t)
        d = np.inf
        if idx > 0:       d = min(d, abs(t - ref_times[idx - 1]))
        if idx < len(ref_times): d = min(d, abs(t - ref_times[idx]))
        dists.append(d)
    return np.array(dists)


def visualize_alignment(audio_path, beatmap_paths, output_path, bpm=120.0, offset=0.0, tol_ms=50.0):
    lines = []

    def p(s=""): lines.append(s); print(s)

    p(f"Audio    : {os.path.basename(audio_path)}")
    p(f"Tolerance: {tol_ms:.0f} ms")
    p("=" * 56)

    # Detect audio features once
    onset_times, sr = detect_onsets(audio_path)
    beat_times, tempo = detect_beats(audio_path, start_bpm=bpm)

    y, _ = librosa.load(audio_path, sr=sr)
    y_perc = librosa.effects.hpss(y)[1]
    perc_frames = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=True)
    perc_times = librosa.frames_to_time(perc_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    p(f"Onsets   : {len(onset_times)}")
    p(f"Beats    : {len(beat_times)}  (tempo: {tempo:.1f} BPM)")
    p(f"Duration : {duration:.1f}s")
    p()

    tol = tol_ms / 1000.0

    for bp_path in beatmap_paths:
        if not os.path.exists(bp_path):
            p(f"Missing  : {bp_path}")
            continue

        name = os.path.basename(bp_path)
        p("-" * 56)
        p(f"Beatmap  : {name}")

        measures = load_beatmap_measures(bp_path, "Hard")
        step_times, total_arrows, _, _, _ = get_step_times(measures, bpm, offset)

        if len(step_times) == 0:
            p("  (no steps found)")
            p()
            continue

        valid = step_times[step_times <= duration]
        oob   = len(step_times) - len(valid)

        d_onset = _nearest(valid, onset_times)
        d_beat  = _nearest(valid, beat_times)
        d_perc  = _nearest(valid, perc_times)

        n = len(valid)
        on_onset = int((d_onset <= tol).sum())
        on_beat  = int((d_beat  <= tol).sum())
        on_perc  = int((d_perc  <= tol).sum())

        p(f"  Steps (valid)    : {n}  |  arrows: {total_arrows}" + (f"  |  out-of-bounds: {oob}" if oob else ""))
        p(f"  Onset alignment  : {on_onset}/{n}  →  {on_onset/n*100:.1f}%")
        p(f"  Beat  alignment  : {on_beat}/{n}  →  {on_beat/n*100:.1f}%")
        p(f"  Perc. alignment  : {on_perc}/{n}  →  {on_perc/n*100:.1f}%")
        p(f"  Mean dist onset  : {np.mean(d_onset)*1000:.1f} ms")
        p(f"  Mean dist beat   : {np.mean(d_beat)*1000:.1f} ms")
        p()

    p("=" * 56)

    if output_path:
        out = os.path.splitext(output_path)[0] + ".txt"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved to: {out}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AI Beatmap Alignment")
    parser.add_argument("--audio", required=True, help="Path to original audio (.ogg/.mp3)")
    parser.add_argument("--beatmaps", nargs='*', help="Paths to generated beatmap files")
    parser.add_argument("--pattern", type=str, help="Substring to find in filename (e.g. task0003)")
    parser.add_argument("--output", default="alignment_compare.html", help="HTML output file name")
    parser.add_argument("--bpm", type=float, default=120.0, help="Fallback BPM if metadata is missing")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset in seconds")
    
    args = parser.parse_args()
    
    beatmaps = args.beatmaps or []
    if args.pattern:
        song_dir = os.path.dirname(os.path.abspath(args.audio))
        for f in os.listdir(song_dir):
            path = os.path.join(song_dir, f)
            # Find original chart
            if f.endswith(('.ssc', '.sm')) and path not in beatmaps:
                beatmaps.append(path)
            # Find matching generated txts
            elif args.pattern in f and f.endswith('.txt') and path not in beatmaps:
                beatmaps.append(path)
                
    if not beatmaps:
        print("Error: No beatmaps provided or found matching pattern.")
        sys.exit(1)
        
    visualize_alignment(args.audio, beatmaps, args.output, args.bpm, args.offset)
