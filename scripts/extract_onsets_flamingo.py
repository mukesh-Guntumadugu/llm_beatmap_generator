#!/usr/bin/env python3
"""
extract_onsets_flamingo.py
==========================
Extracts raw onsets (in ms) using Music-Flamingo for all songs.
Loops through the 20 test songs, chunks audio into 5s windows, 
and queries the Flamingo model directly.
"""

import os
import sys
import csv
import glob
import re
import gc
import tempfile
import torch
import numpy as np
import librosa
import soundfile as sf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR = os.path.join(_PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
os.environ["HF_HOME"] = os.path.join(_PROJECT_ROOT, "Music-Flamingo", "checkpoints")

from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo

def extract_for_song(song_dir, chunk_sec=5.0):
    audio_files = (glob.glob(os.path.join(song_dir, "*.ogg")) +
                   glob.glob(os.path.join(song_dir, "*.mp3")) +
                   glob.glob(os.path.join(song_dir, "*.wav")))
    if not audio_files:
        return []

    audio_path = audio_files[0]
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    all_onsets_ms = []

    for start in np.arange(0, duration, chunk_sec):
        end = min(start + chunk_sec, duration)
        if end - start < 4.9:
            continue

        chunk = y[int(start*sr):int(end*sr)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, sr)
            tmp_path = tmp.name

        prompt = f"List all the onset timestamps in this {round(end-start,1)}s audio clip in milliseconds."
        response = generate_beatmap_with_flamingo(tmp_path, prompt)

        nums = re.findall(r"\d+\.?\d*", str(response))
        for n in nums:
            all_onsets_ms.append(int(round(float(n) + start*1000)))

        os.remove(tmp_path)
        gc.collect(); torch.cuda.empty_cache()

    return sorted(all_onsets_ms)

def main():
    print("Loading Music-Flamingo model (~30GB)...", flush=True)
    setup_music_flamingo()

    song_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])
    
    for song_dir in song_dirs:
        song_name = os.path.basename(song_dir)
        print(f"\nProcessing {song_name}...")
        
        onsets_ms = extract_for_song(song_dir, chunk_sec=5.0)
        if not onsets_ms:
            continue

        out_dir = os.path.join(song_dir, "flamingo_onsets")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = song_name.replace(" ", "_").replace("'", "")
        out_path = os.path.join(out_dir, f"{safe_name}_Flamingo_5s.txt")

        with open(out_path, "w") as f:
            for ms in onsets_ms:
                f.write(f"{ms}\n")

        print(f"  ✅ Saved {len(onsets_ms)} onsets → {out_path}", flush=True)

if __name__ == "__main__":
    main()
