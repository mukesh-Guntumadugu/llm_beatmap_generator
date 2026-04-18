#!/usr/bin/env python3
"""
Onset Extractor — uses a local Music-Flamingo model to extract
raw onset timestamps (ms) for every song in the Fraxtil pack.

Outputs a standard .csv format with chronological onsets.

Usage:
    python3 scripts/extract_onsets_flamingo.py --chunk_sec 20
    python3 scripts/extract_onsets_flamingo.py --chunk_sec 10 --songs "Bad Ketchup"
"""

import os
import sys
import csv
import glob
import re
import time
import datetime
import argparse
import tempfile
import gc
import torch
import librosa
import soundfile as sf
import numpy as np

# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR = os.path.join(
    _PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# Connect to Flamingo model
os.environ["HF_HOME"] = os.path.join(_PROJECT_ROOT, "Music-Flamingo", "checkpoints")

try:
    from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
except ImportError:
    print("ERROR Could not import src.music_flamingo_interface. Check paths.")
    sys.exit(1)


def is_hallucinated_loop(onsets: list[int]) -> bool:
    if not onsets or len(onsets) < 10:
        return False
    if len(onsets) > 150:
        return True 
    times = onsets
    deltas = [round(times[j+1] - times[j], 1) for j in range(len(times)-1)]
    for j in range(len(deltas) - 8):
        window = deltas[j:j+8]
        if all(w == window[0] for w in window):
            return True
    return False

def parse_flamingo_onsets(response: str, start_sec: float) -> list[int]:
    """Parse output numbers from anchored format in seconds, relative to chunk start."""
    text_clean = re.sub(r"```[\w]*\n?", "", response or "").strip()
    onsets = []
    
    # 1. Anti-Hallucination Secure Anchored Parsing (Seconds)
    found_matches = re.findall(r':\s*([0-9]+\.?[0-9]*)', text_clean)
    if found_matches:
        for val in found_matches:
            try:
                ms = float(val) * 1000.0
                if 0.0 <= ms <= 600_000:
                    onsets.append(int(round(ms + (start_sec * 1000))))
            except ValueError:
                pass
        if onsets:
            return sorted(set(onsets))

    # 2. Fallback to naive float searching if formatting rule ignored
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text_clean)
    for n in nums:
        try:
            val_f = float(n)
            # Assume < 1000 is seconds to be safe
            ms = val_f * 1000.0 if val_f < 600.0 else val_f
            if 0.0 <= ms <= 600_000:
                onsets.append(int(round(ms + (start_sec * 1000))))
        except ValueError:
            pass
            
    return sorted(set(onsets))


def extract_flamingo_onsets_for_song(song_dir: str, chunk_sec: float) -> tuple[list[int], float]:
    """Slice audio into chunk_sec chunks, query local Flamingo, return absolute ms timestamps."""
    audio_files = (glob.glob(os.path.join(song_dir, "*.ogg")) +
                   glob.glob(os.path.join(song_dir, "*.mp3")) +
                   glob.glob(os.path.join(song_dir, "*.wav")))
    
    if not audio_files:
        print(f"   WARNING No audio file found — skipping.")
        return [], 0.0

    audio_path = audio_files[0]
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    all_onsets_ms = []
    total_request_time = 0.0
    chunk_starts = np.arange(0, duration, chunk_sec)

    for chunk_idx, start_sec in enumerate(chunk_starts):
        end_sec = min(start_sec + chunk_sec, duration)
        
        # Skip weird final empty chunks
        if end_sec - start_sec < 0.5:
            continue

        start_frame = int(start_sec * sr)
        end_frame   = int(end_sec   * sr)
        y_chunk     = y[start_frame:end_frame]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y_chunk, sr)
            tmp_path = tmp.name

        prompt = (
            f"<Audio>\nPlease listen to the audio and extract the specific musical burst onsets from this {round(end_sec-start_sec, 1)}-second clip. "
            "You must follow these strict rules to prevent hallucinations:\n"
            "1. Do not give me a bare list of numbers. Format each line exactly as: [Hit] : [Onset Timestamp in seconds]\n"
            "2. CRITICAL: Do not interpolate, guess, auto-fill, or count up sequentially. Only output the exact numbers found in the audio.\n"
            "3. Do not make up extra data if the audio is sparse. Stop generating when you reach the end."
        )

        max_retries = 3
        chunk_onsets_ms = []
        req_time = 0.0

        for attempt in range(max_retries):
            try:
                req_start = time.time()
                # Local Inference!
                response = generate_beatmap_with_flamingo(tmp_path, prompt)
                req_time = time.time() - req_start
                total_request_time += req_time
                
                chunk_onsets_ms = parse_flamingo_onsets(response, start_sec)
                
                if is_hallucinated_loop(chunk_onsets_ms):
                    print(f"    [!] HALLUCINATION CAUGHT: Neural counting drift detected! Retrying chunk {chunk_idx}...")
                    chunk_onsets_ms = []
                    continue
                    
                break # Success! Break the retry loop
                
            except Exception as e:
                print(f"  [!] Chunk {chunk_idx} inference failed: {e}")
                if attempt == max_retries - 1:
                    chunk_onsets_ms = []
                    break
        all_onsets_ms.extend(chunk_onsets_ms)

        print(f"  Chunk {chunk_idx+1}/{len(chunk_starts)}: {len(chunk_onsets_ms)} onsets detected (inference time: {req_time:.2f}s)")

        # Cleanup memory from local inference
        os.remove(tmp_path)
        gc.collect()
        torch.cuda.empty_cache()

    return sorted(list(set(all_onsets_ms))), total_request_time


def main():
    parser = argparse.ArgumentParser(description="Extract onsets using local Flamingo model")
    parser.add_argument("--chunk_sec", type=float, default=15.0, help="Chunk size in seconds")
    parser.add_argument("--songs", nargs="+", default=None,
                        help="Specific song names to process (default: all songs)")
    args = parser.parse_args()

    # Determine targets
    if args.songs:
        song_dirs = [os.path.join(BASE_DIR, s) for s in args.songs if os.path.isdir(os.path.join(BASE_DIR, s))]
    else:
        song_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])

    total = len(song_dirs)
    if total == 0:
        print("No valid directories found in Fraxtil packs.")
        return

    print(f"Loading local Music-Flamingo model to GPU (~30GB) - this might take a minute...")
    setup_music_flamingo()
    print("Model Active.\n")

    print(f"Processing {total} songs with chunk size {args.chunk_sec}s...")

    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        print(f"[{idx+1}/{total}] {song_name}")

        onsets_ms, inference_time = extract_flamingo_onsets_for_song(song_dir, args.chunk_sec)
        
        if not onsets_ms:
            print(f"  WARNING No onsets extracted for {song_name}")
            continue

        # Save output
        out_dir = os.path.join(song_dir, "flamingo_onsets")
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = song_name.replace(" ", "_").replace("'", "")
        out_path = os.path.join(out_dir, f"{safe_name}_Flamingo_{int(args.chunk_sec)}s_{timestamp}.csv")

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["onset_ms"])
            for ms in onsets_ms:
                w.writerow([ms])

        print(f"  -> Saved {len(onsets_ms)} onsets -> flamingo_onsets/{os.path.basename(out_path)}")
        print(f"  -> Total inference time for song: {inference_time:.2f}s\n")

    print("All Flamingo extraction subsets finished!")


if __name__ == "__main__":
    main()
