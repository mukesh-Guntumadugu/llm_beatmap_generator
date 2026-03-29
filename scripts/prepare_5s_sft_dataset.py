#!/usr/bin/env python3
"""
prepare_5s_sft_dataset.py
=========================
Slices full-length tracks into strict 5-second sequential chunks,
shifts onset ground truth into those local 5s windows, and builds a generic JSONL 
dataset perfectly calibrated for multi-model SFT (MuMu, Flamingo, DeepResonance).
"""

import os
import json
import csv
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm

# ── Configuration ──
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "pixabay_music")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sft_dataset_5s_chunks")
AUDIO_OUT_DIR = os.path.join(OUTPUT_DIR, "audio")
CHUNK_SIZE_SEC = 5.0
HPC_BASE_PATH = "/data/mg546924/llm_beatmap_generator/sft_dataset_5s_chunks/audio"

def load_onsets(csv_path):
    """Load onsets in seconds from the original Librosa CSV."""
    onsets_sec = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if 'onset_sec' in row:
                    onsets_sec.append(float(row['onset_sec']))
                elif 'onset_ms' in row:
                    onsets_sec.append(float(row['onset_ms']) / 1000.0)
                else: 
                    val = float(list(row.values())[0])
                    onsets_sec.append(val / 1000.0 if val > 1000 else val)
            except ValueError:
                pass
    return sorted(onsets_sec)

def process_song(audio_path, onsets_sec, jsonl_file):
    """Chop song into 5s windows and write prompt/response datasets."""
    song_basename = os.path.splitext(os.path.basename(audio_path))[0]
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Skipping {song_basename}: {e}")
        return 0
        
    duration = librosa.get_duration(y=y, sr=sr)
    chunks_created = 0
    
    for i, start_time in enumerate(np.arange(0, duration, CHUNK_SIZE_SEC)):
        end_time = min(start_time + CHUNK_SIZE_SEC, duration)
        
        # Enforce exact 5.0s length to keep GPU VRAM batches perfectly uniform
        if end_time - start_time < (CHUNK_SIZE_SEC - 0.1): 
            continue
            
        start_samp = int(start_time * sr)
        end_samp = int(end_time * sr)
        chunk_y = y[start_samp:end_samp]
        
        if librosa.feature.rms(y=chunk_y).mean() < 0.005:
            continue # skip literal silence
            
        chunk_filename = f"{song_basename}_chunk{i:03d}.wav"
        local_out_path = os.path.join(AUDIO_OUT_DIR, chunk_filename)
        sf.write(local_out_path, chunk_y, sr)
        
        # Shift onsets purely into the 0.0 to 5.0 local chunk window
        local_onsets_ms = []
        for o_sec in onsets_sec:
            if start_time <= o_sec < end_time:
                shifted_sec = o_sec - start_time
                local_onsets_ms.append(int(round(shifted_sec * 1000)))
        
        prompt = (
            "You are a rhythm game beatmap generator. "
            f"This is a {round(end_time-start_time, 1)}s audio clip. "
            "List ONLY the onset timestamps in milliseconds as plain numbers separated by commas."
        )
        
        response = ", ".join(map(str, local_onsets_ms))
        if not response: 
            response = "0"
            
        hpc_audio_path = f"{HPC_BASE_PATH}/{chunk_filename}"
        
        data_entry = {
            "id": f"{song_basename}_{i:03d}",
            "audio_path": hpc_audio_path, # Remote HPC path perfectly assigned
            "duration": round(end_time-start_time, 2),
            "text_prompt": prompt,
            "text_response": response,
            "beat_count": len(local_onsets_ms)
        }
        
        jsonl_file.write(json.dumps(data_entry) + "\n")
        chunks_created += 1
        
    return chunks_created

def main():
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    
    audio_files = []
    for ext in ["*.mp3", "*.ogg", "*.wav"]:
        audio_files.extend(glob(os.path.join(AUDIO_DIR, ext)))
        
    print(f"Found {len(audio_files)} audio files. Beginning 5-second chunking.")
    
    total_chunks = 0
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for audio_path in tqdm(audio_files):
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Find matching ground truth onsets from Librosa Step B
            csv_matches = glob(os.path.join(AUDIO_DIR, f"original_onsets_{basename}*.csv"))
            if not csv_matches:
                continue
                
            onsets_sec = load_onsets(csv_matches[0])
            if not onsets_sec:
                continue
                
            total_chunks += process_song(audio_path, onsets_sec, f_out)
            
    print(f"\n✅ DATASET CREATED: {total_chunks} perfectly formatted 5.0-second training samples!")
    print(f"Dataset path: {jsonl_path}")

if __name__ == "__main__":
    main()
