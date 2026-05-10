#!/usr/bin/env python3
"""
prepare_bpm_sft_dataset.py
==========================
Generates a JSONL dataset for Qwen2-Audio to learn BPM extraction.
Reads the primary BPM from `local_bpm_analysis.json` and pairs it with 
the first 30 seconds of audio.
"""

import os
import glob
import json
import librosa
import soundfile as sf
import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUTPUT_DIR = os.path.join(PROJ_DIR, "bpm_sft_dataset")
AUDIO_OUT_DIR = os.path.join(OUTPUT_DIR, "audio")
BPM_JSON = os.path.join(PROJ_DIR, "local_bpm_analysis.json")

CHUNK_DURATION = 30.0  # seconds to give model enough context

def main():
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    jsonl_data = []
    
    if not os.path.exists(BPM_JSON):
        print(f"Error: {BPM_JSON} not found!")
        return

    with open(BPM_JSON, "r") as f:
        bpm_db = json.load(f)

    # bpm_db format: {"song.sm": {"file": "...", "sections": [{"bpm": 145.0, ...}]}}
    
    # Map song dir name to its BPM
    song_to_bpm = {}
    for sm_key, sm_data in bpm_db.items():
        if "sections" in sm_data and len(sm_data["sections"]) > 0:
            # Grab the primary BPM (the first one)
            primary_bpm = sm_data["sections"][0]["bpm"]
            # The sm_key is often just "Song Name.sm". Strip .sm
            song_name = sm_key.replace(".sm", "").replace(".ssc", "")
            song_to_bpm[song_name] = primary_bpm

    # Iterate through dataset directory
    song_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith("_")]
    
    for song in song_dirs:
        # Match song name
        matched_bpm = None
        for key in song_to_bpm.keys():
            if key in song or song in key:
                matched_bpm = song_to_bpm[key]
                break
                
        if not matched_bpm:
            print(f"  Skipping {song}: No BPM found in DB.")
            continue
            
        print(f"Processing {song} | BPM: {matched_bpm}")
        song_dir = os.path.join(DATASET_DIR, song)
            
        # Find audio
        audio_files = glob.glob(os.path.join(song_dir, "*.ogg")) + glob.glob(os.path.join(song_dir, "*.mp3"))
        if not audio_files:
            continue
        audio_path = audio_files[0]
        
        # Load audio and chunk
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            end_frame = int(CHUNK_DURATION * sr)
            y_chunk = y[:end_frame]
            
            chunk_filename = f"{song.replace(' ', '_')}_bpm.wav"
            chunk_out_path = os.path.join(AUDIO_OUT_DIR, chunk_filename)
            sf.write(chunk_out_path, y_chunk, sr)
            
            # Create Qwen2-Audio JSONL entry
            entry = {
                "id": f"{song.replace(' ', '_')}_bpm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": os.path.abspath(chunk_out_path)},
                            {"type": "text", "text": "What is the exact mathematical BPM (Beats Per Minute) of this audio clip?"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"The BPM is {matched_bpm}"}
                        ]
                    }
                ]
            }
            jsonl_data.append(entry)
        except Exception as e:
            print(f"  Error processing {song}: {e}")
            
    # Save JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "bpm_dataset.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\\n")
            
    print(f"\\nCreated {len(jsonl_data)} audio-BPM pairs for SFT.")
    print(f"Dataset saved to {jsonl_path}")

if __name__ == "__main__":
    main()
