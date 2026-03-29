#!/usr/bin/env python3
"""
benchmark_deepresonance_bpm.py
==============================
Benchmarks BPM (Beats Per Minute) detection mathematically using Librosa
against natural-language BPM analysis using Sony DeepResonance.
Outputs results to a CSV file.
"""

import os
import sys
import csv
import torch
import librosa
import warnings
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- Bypass DeepSpeed NVCC Bug ---
from unittest.mock import MagicMock
try:
    import triton
except ImportError:
    sys.modules['triton'] = MagicMock()
sys.modules['triton.ops'] = MagicMock()
sys.modules['triton.ops.matmul_perf_model'] = MagicMock()
# ---------------------------------

# ── Configuration ──
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "pixabay_music")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "deepresonance_bpm_benchmark.csv")
DEEPRESONANCE_DIR = os.path.join(os.path.dirname(__file__), "..", "DeepResonance", "code")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "DeepResonance", "ckpt")

# Limit the LLM context to the first 30 seconds for speed and memory efficiency
EVAL_DURATION_SEC = 30.0

def load_deepresonance():
    """Initializes the DeepResonance multimodal predictor."""
    sys.path.append(DEEPRESONANCE_DIR)
    os.chdir(DEEPRESONANCE_DIR)
    from inference_deepresonance import DeepResonancePredict
    
    print("Loading DeepResonance 12GB checkpoint into VRAM...")
    model_args = {
        'stage': 2,
        'mode': 'test',
        'project_path': DEEPRESONANCE_DIR,
        'dataset': 'musiccaps',
        'llm_path': os.path.join(CKPT_PATH, 'pretrained_ckpt', 'vicuna_ckpt', '7b_v0'),
        'imagebind_path': os.path.join(CKPT_PATH, 'pretrained_ckpt', 'imagebind_ckpt', 'huge'),
        'imagebind_version': 'huge',
        'max_length': 512,
        'max_output_length': 512,
        'num_clip_tokens': 77,
        'gen_emb_dim': 768,
        'preencoding_dropout': 0.1,
        'num_preencoding_layers': 1,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'freeze_lm': False,
        'freeze_input_proj': False,
        'freeze_output_proj': False,
        'prompt': '',
        'prellmfusion': True,
        'prellmfusion_dropout': 0.1,
        'num_prellmfusion_layers': 1,
        'imagebind_embs_seq': True,
        'topp': 1.0,
        'temp': 0.1,
        'ckpt_path': os.path.join(CKPT_PATH, 'DeepResonance_data_models', 'ckpt',
                                  'deepresonance_beta_delta_ckpt', 'delta_ckpt',
                                  'deepresonance', '7b_tiva_v0'),
    }
    return DeepResonancePredict(model_args)

def extract_librosa_bpms(y, sr):
    """
    Computes mathematical BPM. Since songs can have dynamically shifting tempos, 
    we capture the primary tempos calculated over the track's duration.
    """
    # Extract dynamic tempo array
    tempo_array = librosa.feature.tempo(y=y, sr=sr, aggregate=None)
    
    # We round the array to integer BPMs to cluster them
    rounded_tempos = np.round(tempo_array).astype(int)
    
    # Find the most uniquely occurring tempos taking up at least 10% of the song
    unique_tempos, counts = np.unique(rounded_tempos, return_counts=True)
    significant_tempos = []
    threshold = 0.10 * len(tempo_array)
    
    for tempo_val, count in zip(unique_tempos, counts):
        if count >= threshold:
            significant_tempos.append(tempo_val)
            
    # Fallback to single static tempo if strict alignment fails
    if not significant_tempos:
        static_tempo = librosa.feature.tempo(y=y, sr=sr)[0]
        significant_tempos = [int(round(static_tempo))]
        
    return sorted(significant_tempos)

def predict_dr_bpm(dr_predictor, eval_audio_path):
    """Prompts DeepResonance for the BPM of an audio clip."""
    prompt = (
        "<Audio>\n"
        "What is the exact BPM (Beats Per Minute) and tempo of this song? "
        "If the song has multiple tempo changes, list all of them. "
        "Be extremely concise and only output the numbers."
    )
    
    inputs = {
        "inputs": [prompt],
        "instructions": [prompt],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(eval_audio_path)]],
        "mm_root_path": os.path.dirname(eval_audio_path),
        "outputs": [""],
    }
    
    try:
        response = dr_predictor.predict(
            inputs,
            max_tgt_len=1024,
            top_p=1.0,
            temperature=0.1,
            stops_id=[[835]],
        )
        if isinstance(response, list):
            response = response[0].split("\n###")[0]
        return response.strip()
    except Exception as e:
        import traceback
        tb = traceback.format_exc().replace("\n", " -- ")
        return f"ERROR: {str(e)} | TRACE: {tb[-1000:]}"

def process_files(audio_files, dr_predictor):
    # Resume safety: If CSV exists, skip already processed files
    processed = set()
    file_mode = "w"
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    processed.add(row[0])
        file_mode = "a"

    with open(OUTPUT_CSV, file_mode, newline="") as f_out:
        writer = csv.writer(f_out)
        if file_mode == "w":
            writer.writerow(["song_name", "librosa_actual_bpm", "deepresonance_predicted_bpm"])
            
        for audio_path in tqdm(audio_files, desc="Evaluating BPMs"):
            basename = os.path.basename(audio_path)
            if basename in processed:
                continue

            try:
                # 1. Load full track for Librosa 
                y_full, sr_full = librosa.load(audio_path, sr=16000)
                actual_bpms = extract_librosa_bpms(y_full, sr_full)
                
                # Format to string, e.g. "120" or "[120, 140]"
                if len(actual_bpms) == 1:
                    actual_bpms_str = str(actual_bpms[0])
                else:
                    actual_bpms_str = str(actual_bpms)

                # 2. Extract first 30s chunk to prevent LLM memory overflow/latency
                samples_to_keep = int(sr_full * EVAL_DURATION_SEC)
                y_clip = y_full[:samples_to_keep]
                tmp_clip_path = "/tmp/dr_bpm_eval_clip_fixed.wav"
                sf.write(tmp_clip_path, y_clip, sr_full, subtype='PCM_16')

                # 3. Query DeepResonance
                predicted_bpm = predict_dr_bpm(dr_predictor, tmp_clip_path)

                # 4. Write row directly to disk
                writer.writerow([basename, actual_bpms_str, predicted_bpm])
                f_out.flush()
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc().replace('\n', ' -- ')
                print(f"\nSkipping {basename} due to error: {e}")
                writer.writerow([basename, "ERROR", f"ERROR: {e} | {tb}"])
                f_out.flush()

def main():
    print(f"Finding audio files in {AUDIO_DIR}...")
    audio_files = []
    for ext in ["*.mp3", "*.ogg", "*.wav"]:
        audio_files.extend(glob(os.path.join(AUDIO_DIR, ext)))
        
    if not audio_files:
        print("No audio files found! Exiting.")
        return

    print(f"Found {len(audio_files)} songs to benchmark.")
    
    # Initialize Model First
    dr_predictor = load_deepresonance()
    print("✅ Inference Model ready.")
    
    # Run evaluation on only 10 songs for fast testing
    process_files(audio_files[:10], dr_predictor)
    print(f"\n✅ Benchmarking Complete! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
