#!/usr/bin/env python3
"""
extract_onsets_mumu.py
======================
Extracts raw onsets (in ms) using MuMu-LLaMA for all songs.
Loops through the 20 test songs, chunks audio into 5s windows, 
and queries the MuMu model directly.
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
import datetime
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR = os.path.join(_PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")

from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu

def extract_for_song(song_dir, chunk_sec=5.0):
    audio_files = (glob.glob(os.path.join(song_dir, "*.ogg")) +
                   glob.glob(os.path.join(song_dir, "*.mp3")) +
                   glob.glob(os.path.join(song_dir, "*.wav")))
    if not audio_files:
        return []

    audio_path = audio_files[0]
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
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
        response = generate_beatmap_with_mumu(tmp_path, prompt)

        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", response or "")
        for n in nums:
            all_onsets_ms.append(int(round(float(n) + start*1000)))

        os.remove(tmp_path)
        gc.collect(); torch.cuda.empty_cache()

    return sorted(all_onsets_ms)

def main():
    print("Loading MuMu-LLaMA model...", flush=True)
    setup_mumu()

    song_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])
    
    for song_dir in tqdm(song_dirs, desc="Processing Songs with MuMu"):
        song_name = os.path.basename(song_dir)
        print(f"\nProcessing {song_name}...")
        
        onsets_ms = extract_for_song(song_dir, chunk_sec=5.0)
        if not onsets_ms:
            continue

        out_dir = os.path.join(song_dir, "mumu_onsets")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = song_name.replace(" ", "_").replace("'", "")
        out_path = os.path.join(out_dir, f"{safe_name}_MuMu_5s.txt")

        with open(out_path, "w") as f:
            for ms in onsets_ms:
                f.write(f"{ms}\n")

        print(f"  ✅ Saved {len(onsets_ms)} onsets → {out_path}", flush=True)

if __name__ == "__main__":
    main()
