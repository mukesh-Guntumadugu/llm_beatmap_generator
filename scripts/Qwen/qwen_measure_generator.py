#!/usr/bin/env python3
import os
import sys
import tempfile
import gc
import csv
import datetime
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Connect up to main project path
PROJ = os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))

try:
    from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
except ImportError:
    print("Could not import qwen_interface. Ensure BENCHMARK_PROJ is correct.")
    sys.exit(1)

AUDIO_PATH = os.environ.get("BENCHMARK_AUDIO", "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg")
OUT_DIR = os.environ.get("BENCHMARK_OUT", os.path.dirname(AUDIO_PATH))

# Valid 15 combinations without 0000 (as user requested)
VALID_STEP_COMBOS = [
    "0001", "0010", "0100", "1000",          # Singles
    "0011", "0101", "0110", "1001", "1010", "1100", # Jumps
    "0111", "1011", "1101", "1110",          # Hands
    "1111"                                   # Quads
]

def main():
    print(f"Loading Qwen model...")
    setup_qwen()

    print("Loading audio and analyzing onsets via Librosa...")
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract BPM and raw onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    global_bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    print(f"Global BPM detected: {global_bpm:.1f}")

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Math for Beginner grid (4 rows per measure)
    beats_per_measure = 4
    beat_duration_sec = 60.0 / global_bpm
    measure_duration_sec = beat_duration_sec * beats_per_measure
    total_measures = int(np.ceil(duration / measure_duration_sec))

    print(f"Total Song Duration: {duration:.2f}s")
    print(f"Measure Duration: {measure_duration_sec:.3f}s")
    print(f"Total Measures to generate (Beginner 4-rows/measure): {total_measures}")
    print("--------------------------------------------------")

    history_log = []
    out_rows = []

    for M in tqdm(range(total_measures), desc="Generating Measures"):
        start_time = M * measure_duration_sec
        end_time = start_time + measure_duration_sec
        
        if start_time >= duration: 
            break
        
        # 1. Chunk audio for this specific measure
        chunk_y = y[int(start_time*sr):int(min(end_time, duration)*sr)]
        
        # Pad chunk explicitly to ensure the model always hears exactly exactly that measure size properly
        target_len = int(measure_duration_sec * sr)
        if len(chunk_y) < target_len:
            pad_len = target_len - len(chunk_y)
            chunk_y = np.pad(chunk_y, (0, pad_len), 'constant')
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk_y, sr)
            tmp_path = tmp.name

        # 2. Gather raw onsets inside this measure window
        m_onsets = [round(t, 2) for t in onset_times if start_time <= t < end_time]
        
        # 3. Build Prompt with exact requirements
        prompt = (
            f"You are generating a StepMania beatmap (Beginner difficulty, 4 rows per measure).\n"
            f"Measure: {M+1} / {total_measures}\n"
            f"Global BPM: {global_bpm:.1f}\n"
            f"This 4/4 measure spans from {start_time:.2f}s to {end_time:.2f}s.\n"
            f"Detected Audio Onsets in this measure window: {m_onsets}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Output EXACTLY 4 strings representing the 4 beats of this measure.\n"
            f"- Each string is 4 characters long (Left, Down, Up, Right).\n"
            f"- LOGIC: If a beat aligns with an onset timestamp, select an appropriate step. If there is NO onset near the beat, you MUST output '0000'.\n"
            f"- Valid step options for an onset (15 options total): {', '.join(VALID_STEP_COMBOS)}\n\n"
            f"HISTORY OF PREVIOUS MEASURE SELECTIONS (Use this to form patterns and avoid repetition):\n"
        )
        
        # Full history payload
        if len(history_log) > 0:
            hist_text = "\n".join(history_log)
        else:
            hist_text = "None (this is the first measure)."
            
        prompt += hist_text + "\n\n---> Please output only the 4 rows for THIS measure."
        
        # 4. Generate Output
        response = generate_beatmap_with_qwen(tmp_path, prompt)
        
        # Parse only valid rows (ignoring conversational filler if hallucinated)
        lines = [L.strip() for L in (response or "").splitlines() if len(L.strip()) == 4 and all(c in '01234' for c in L.strip())]
        
        # Enforce 4 rows strictly
        while len(lines) < 4: lines.append("0000")
        if len(lines) > 4: lines = lines[:4]
        
        measure_out_str = f"Measure {M+1}:\n" + "\n".join(lines)
        history_log.append(measure_out_str)
        out_rows.extend(lines)
        
        # Cleanup memory for loop 
        os.remove(tmp_path)
        gc.collect(); torch.cuda.empty_cache()

    # Write output file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"qwen_MeasureGrid_{ts}.csv")
    txt_path = os.path.join(OUT_DIR, f"qwen_MeasureGrid_{ts}.txt")

    with open(txt_path, "w") as f:
        f.write("\n".join(out_rows))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["notes"])
        for r in out_rows: w.writerow([r])

    print(f"✅ Generated {len(out_rows)} rows across {total_measures} measures.")
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
