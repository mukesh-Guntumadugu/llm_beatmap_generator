#!/usr/bin/env python3
"""
analyze_onset_density.py
========================
Computes onset density statistics across two categories of FULL audio files:

  1. NORMAL SONGS  — full-length normal songs (e.g., from test_dir/Song)
                     This gives us the actual onset count for a regular song.

  2. DDR BEATMAP SONGS — full-length songs in src/musicForBeatmap/ that have
                     associated .sm or .ssc StepMania chart files.
                     These are your actual DDR-style game songs.

For each file the script measures:
  - Total number of onsets detected (via librosa onset_detect)
  - Song duration in seconds
  - Onsets per second (density)

Summary:
  - Avg onset count per full song
  - Avg onset density (onsets/sec)
"""

import os
import sys
import json
import time
import warnings
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ── Suppress noisy librosa/audioread warnings ──────────────────────────────────
warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    print("ERROR: librosa not found. Activate your venv first:")
    print("  source .venv/bin/activate   (or pattern_venv / .myvenv)")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# Normal audio: full songs downloaded from pixabay
NORMAL_AUDIO_DIRS = [
    ROOT / "pixabay_music"
]

# DDR beatmap songs: full-length songs that have .sm/.ssc chart files alongside
BEATMAP_AUDIO_DIR = ROOT / "src" / "musicForBeatmap"

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# Librosa onset detection params (matching your existing pipeline)
SR = 22050          # sample rate
HOP_LENGTH = 512    # hop size in samples
ONSET_UNITS = "time"  # return timestamps in seconds

OUTPUT_JSON = ROOT / "onset_density_results.json"
OUTPUT_NORMAL_CSV  = ROOT / "onset_density_normal_songs.csv"
OUTPUT_BEATMAP_CSV = ROOT / "onset_density_beatmap_songs.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_beatmap_chart(audio_path: Path) -> str:
    """Return the name of the .sm or .ssc file in the same directory, or empty string."""
    d = audio_path.parent
    sm_files = list(d.glob("*.sm"))
    ssc_files = list(d.glob("*.ssc"))
    if ssc_files:
        return ssc_files[0].name
    if sm_files:
        return sm_files[0].name
    return ""

def has_beatmap_chart(audio_path: Path) -> bool:
    return bool(get_beatmap_chart(audio_path))


def collect_audio_files(dirs, require_chart: bool = False) -> List[Path]:
    """
    Walk directories and collect all audio files.
    If require_chart=True, only include files whose folder also has .sm/.ssc.
    """
    files = []
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [SKIP] Directory not found: {d}")
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in AUDIO_EXTS and f.is_file():
                if require_chart and not has_beatmap_chart(f):
                    continue
                files.append(f)
    return files


def analyze_file(path: Path) -> Dict:
    """
    Load audio, detect onsets, return stats dict.
    Returns None on failure.
    """
    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True, duration=None)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 0.5:
            return None  # skip tiny fragments

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=HOP_LENGTH, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

        n_onsets = len(onset_times)
        density = n_onsets / duration if duration > 0 else 0.0  # onsets per second

        return {
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "extension": path.suffix,
            "duration_sec": round(float(duration), 2),
            "n_onsets": int(n_onsets),
            "onsets_per_sec": round(float(density), 4),
            "associated_chart": get_beatmap_chart(path)
        }

    except Exception as e:
        return None  # silently skip corrupt / unreadable files


def process_category(label: str, files: List[Path]) -> Tuple[List[Dict], Dict]:
    """Process all files in a category and return records + summary stats."""
    print(f"\n{'='*60}")
    print(f"  Category : {label}")
    print(f"  Files    : {len(files)}")
    print(f"{'='*60}")

    records = []
    errors = 0
    t0 = time.time()

    for i, path in enumerate(files):
        result = analyze_file(path)
        if result:
            records.append(result)
        else:
            errors += 1

        # Progress every 50 files
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(files)}]  "
                  f"ok={len(records)}  err={errors}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="\r")

    print()  # newline after progress

    if not records:
        print(f"  WARNING: No files successfully analyzed in {label}!")
        return records, {}

    durations     = np.array([r["duration_sec"]    for r in records])
    n_onsets_arr  = np.array([r["n_onsets"]        for r in records])
    densities     = np.array([r["onsets_per_sec"]  for r in records])

    summary = {
        "label":               label,
        "total_files_found":   len(files),
        "total_files_ok":      len(records),
        "total_files_error":   errors,
        # Duration
        "avg_duration_sec":    round(float(durations.mean()), 2),
        "median_duration_sec": round(float(np.median(durations)), 2),
        # Onset count
        "avg_n_onsets":        round(float(n_onsets_arr.mean()), 2),
        "median_n_onsets":     round(float(np.median(n_onsets_arr)), 2),
        "min_n_onsets":        int(n_onsets_arr.min()),
        "max_n_onsets":        int(n_onsets_arr.max()),
        # Onset density (onsets/sec)
        "avg_density_onsets_per_sec":    round(float(densities.mean()), 4),
        "median_density_onsets_per_sec": round(float(np.median(densities)), 4),
        "std_density_onsets_per_sec":    round(float(densities.std()), 4),
        "min_density_onsets_per_sec":    round(float(densities.min()), 4),
        "max_density_onsets_per_sec":    round(float(densities.max()), 4),
    }

    return records, summary


def print_summary(summary: Dict):
    if not summary:
        return
    label = summary['label']
    print(f"\n  ── {label} ──")
    print(f"  Files analyzed          : {summary['total_files_ok']} / {summary['total_files_found']}")
    print(f"  Avg file duration       : {summary['avg_duration_sec']}s  (median {summary['median_duration_sec']}s)")
    print(f"  Avg onsets per song     : {summary['avg_n_onsets']} onsets/song  ← TOTAL ONSETS")
    print(f"  Avg onset density       : {summary['avg_density_onsets_per_sec']} onsets/sec")
    print(f"  Density median/std      : {summary['median_density_onsets_per_sec']} / {summary['std_density_onsets_per_sec']} onsets/sec")
    print(f"  Density range           : {summary['min_density_onsets_per_sec']} – {summary['max_density_onsets_per_sec']} onsets/sec")


def save_csv(records: List[Dict], path: Path):
    if not records:
        return
    import csv
    fieldnames = ["file_path", "file_name", "extension", "duration_sec", "n_onsets", "onsets_per_sec", "associated_chart"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"  Detailed CSV saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ONSET DENSITY ANALYSIS: FULL SONGS")
    print("  Comparing: Normal full songs vs DDR Beatmap songs")
    print("="*60)

    # ── 1. Normal songs (full length) ──────────────────────────────────────
    print("\n[NOTE] Processing normal full-length songs.")
    normal_files = collect_audio_files(NORMAL_AUDIO_DIRS, require_chart=False)
    normal_files = list({f.resolve(): f for f in normal_files}.values())
    print(f"Normal full songs found : {len(normal_files)}")

    normal_records, normal_summary = process_category("NORMAL FULL SONGS", normal_files)

    # ── 2. DDR beatmap songs (full-length, with .sm/.ssc charts) ──────────────
    print("\n[NOTE] Processing DDR beatmap songs (must have .sm/.ssc file).\n")
    beatmap_files = collect_audio_files(BEATMAP_AUDIO_DIR, require_chart=True)
    beatmap_files = list({f.resolve(): f for f in beatmap_files}.values())
    print(f"DDR beatmap songs found : {len(beatmap_files)}")

    beatmap_records, beatmap_summary = process_category("DDR BEATMAP SONGS", beatmap_files)

    # ── 3. Print comparison ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)

    print_summary(normal_summary)
    print_summary(beatmap_summary)

    if normal_summary and beatmap_summary:
        n_onsets = normal_summary["avg_n_onsets"]
        b_onsets = beatmap_summary["avg_n_onsets"]
        diff_onsets = b_onsets - n_onsets
        ratio_onsets = b_onsets / n_onsets if n_onsets > 0 else float("nan")

        print(f"\n{'='*60}")
        print(f"  KEY FINDING: TOTAL ONSETS PER SONG")
        print(f"{'='*60}")
        print(f"  Normal song avg onsets   : {n_onsets:.0f} onsets per song")
        print(f"  DDR beatmap avg onsets   : {b_onsets:.0f} onsets per song")
        print(f"  Difference               : {diff_onsets:+.0f} onsets")
        print(f"  DDR / Normal ratio       : {ratio_onsets:.2f}x")

    # ── 4. Save outputs ────────────────────────────────────────────────────────
    result_obj = {
        "normal_songs":       normal_summary,
        "ddr_beatmap_songs":  beatmap_summary,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"\nSummary JSON saved → {OUTPUT_JSON}")

    print()
    save_csv(normal_records, OUTPUT_NORMAL_CSV)
    save_csv(beatmap_records, OUTPUT_BEATMAP_CSV)
    
    print("\nDone.\n")


if __name__ == "__main__":
    main()
