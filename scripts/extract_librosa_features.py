#!/usr/bin/env python3
"""
analyzes a music file (supporting both .mp3 and .wav formats).
The script extracts:
  1. The global BPM of the song.
  2. A dynamic tempo map (array of precise timestamps and corresponding local BPMs/beat frames).
  3. The exact timestamps of all audio onsets.
Outputs to a structured JSON file.
"""
import os
import sys
import json
import argparse
import librosa
import numpy as np

def extract_audio_features(audio_path, output_json=None):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
        
    print(f"Loading '{audio_path}' with Librosa...")
    # Load audio (automatically handles .mp3, .wav, .ogg, etc.)
    y, sr = librosa.load(audio_path, sr=None)
    
    print("Calculating Onsets...")
    # 1. Onsets
    # librosa.onset.onset_detect returns frame indices
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    print("Calculating Tempo and Beat Frames...")
    # 2. Global Tempo and Beats
    # Get global bpm and beat frames
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    global_bpm, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # In recent versions of Librosa, global_bpm is a 1D numpy array with a single float
    if isinstance(global_bpm, np.ndarray):
        global_bpm = float(global_bpm[0])
    else:
        global_bpm = float(global_bpm)
        
    print("Calculating Dynamic Tempo Map...")
    # 3. Dynamic Tempo Map
    # librosa.feature.tempo with aggregate=None returns the local tempo estimated at each frame
    dynamic_tempos = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    
    # To avoid having a massive array (tens of thousands of frames),
    # we'll map the dynamic tempo specifically at the beat timestamps.
    # We find the tempo value exactly where a beat lands.
    tempo_map = []
    for beat_frame, beat_time in zip(beat_frames, beat_times):
        local_bpm = float(dynamic_tempos[beat_frame])
        tempo_map.append({
            "timestamp_sec": round(float(beat_time), 3),
            "local_bpm": round(local_bpm, 2)
        })

    # Prepare final payload
    payload = {
        "audio_file": os.path.basename(audio_path),
        "duration_sec": round(len(y) / sr, 3),
        "global_bpm": round(global_bpm, 2),
        "total_beats": len(beat_times),
        "total_onsets": len(onset_times),
        "onsets_ms": [int(round(float(t) * 1000)) for t in onset_times],
        "tempo_map": tempo_map
    }
    
    # Determine output path
    if not output_json:
        base_name = os.path.splitext(audio_path)[0]
        output_json = f"{base_name}_features.json"
        
    print(f"Writing structured JSON output to '{output_json}'...")
    with open(output_json, 'w') as f:
        json.dump(payload, f, indent=4)
        
    print("Done! ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pure algorithmic DSP features (BPM, Tempo, Onsets) via Librosa.")
    parser.add_argument("audio", help="Path to the audio file (.mp3, .wav, .ogg)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()
    
    extract_audio_features(args.audio, args.output)
