"""
extract_librosa_onsets.py
=========================
Detects audio onsets for all 20 songs in the Fraxtil dataset using librosa
and saves them to CSV files with onset times in milliseconds.

Output filename: original_onsets_<SongName>_<ddmmyyyyHHMMSS>.csv
Output columns : onset_index, onset_ms
"""

import os
import csv
import datetime
import librosa
import numpy as np
from typing import Optional, List

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_audio_file(song_dir: str) -> Optional[str]:
    """Return the first .ogg/.mp3/.wav found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None


def detect_onsets_ms(audio_path: str) -> List[float]:
    """
    Load an audio file and return onset times in milliseconds.
    Uses librosa's onset_detect with backtracking for precision.
    """
    y, sr = librosa.load(audio_path, sr=None)  # keep native sample rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True,   # snap onsets to nearest energy valley
        units="frames"
    )
    onset_times_sec = librosa.frames_to_time(onset_frames, sr=sr)
    onset_ms = [round(float(t) * 1000, 2) for t in onset_times_sec]
    return onset_ms


def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str) -> str:
    """Save onsets to a CSV file and return the file path."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    # Sanitise song name for filesystem
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"original_onsets_{safe_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])

    return filepath


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_") and not d.startswith(".")
    ])

    print(f"Found {len(song_dirs)} song folders in Fraxtil's Arrow Arrangements.\n")
    print(f"{'Song':<45} {'# Onsets':>10}  {'Output file'}")
    print("─" * 110)

    total_songs = 0
    total_onsets = 0

    for song_name in song_dirs:
        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        if audio_path is None:
            print(f"  ⚠️  No audio file found in: {song_name}")
            continue

        try:
            onset_ms = detect_onsets_ms(audio_path)
            out_path = save_onsets_csv(onset_ms, song_name, song_dir)
            rel_out = os.path.relpath(out_path, BASE_DIR)
            print(f"  ✅ {song_name:<43} {len(onset_ms):>10,}  {rel_out}")
            total_songs += 1
            total_onsets += len(onset_ms)
        except Exception as e:
            print(f"  ❌ Error processing '{song_name}': {e}")

    print("─" * 110)
    print(f"\n✅  Processed {total_songs}/{len(song_dirs)} songs  |  Total onsets detected: {total_onsets:,}\n")


if __name__ == "__main__":
    main()
