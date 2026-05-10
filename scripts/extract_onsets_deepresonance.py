#!/usr/bin/env python3
"""
extract_onsets_deepresonance.py
===============================
Extracts raw onsets (in ms) using DeepResonance for all songs.
Loops through the 20 test songs, chunks audio into 5s windows, 
and queries the DeepResonance model directly.
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

# Fix bitsandbytes/CUDA missing libcusparse.so.11
os.environ["LD_LIBRARY_PATH"] = f"/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR = os.path.join(_PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
CKPT = os.path.join(_PROJECT_ROOT, "DeepResonance", "ckpt")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "DeepResonance", "code"))
os.chdir(os.path.join(_PROJECT_ROOT, "DeepResonance", "code"))

from inference_deepresonance import DeepResonancePredict

def extract_for_song(song_dir, model, chunk_sec=5.0):
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

        inputs = {
            "inputs": ["<Audio>"],
            "instructions": [f"List all onset timestamps in milliseconds for this {round(end-start,1)}s clip."],
            "mm_names": [["audio"]],
            "mm_paths": [[os.path.basename(tmp_path)]],
            "mm_root_path": os.path.dirname(tmp_path),
            "outputs": [""],
        }

        resp = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.001, stops_id=[[835]])
        if isinstance(resp, list): resp = resp[0]
        
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", resp or "")
        for n in nums:
            all_onsets_ms.append(int(round(float(n) + start*1000)))

        os.remove(tmp_path)
        gc.collect(); torch.cuda.empty_cache()

    return sorted(all_onsets_ms)

def main():
    print("Loading DeepResonance model...", flush=True)
    args = {
        "stage": 2, "mode": "test", "dataset": "musiccaps",
        "project_path": os.path.join(_PROJECT_ROOT, "DeepResonance", "code"),
        "llm_path": os.path.join(CKPT, "pretrained_ckpt", "vicuna_ckpt", "7b_v0"),
        "imagebind_path": os.path.join(CKPT, "pretrained_ckpt", "imagebind_ckpt", "huge"),
        "imagebind_version": "huge",
        "max_length": 512, "max_output_length": 512,
        "num_clip_tokens": 77, "gen_emb_dim": 768,
        "preencoding_dropout": 0.1, "num_preencoding_layers": 1,
        "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
        "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False,
        "prompt": "", "prellmfusion": True, "prellmfusion_dropout": 0.1,
        "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.001,
        "ckpt_path": os.path.join(CKPT, "DeepResonance_data_models", "ckpt",
                                  "deepresonance_beta_delta_ckpt", "delta_ckpt",
                                  "deepresonance", "7b_tiva_v0"),
    }
    model = DeepResonancePredict(args)

    song_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])
    
    for song_dir in song_dirs:
        song_name = os.path.basename(song_dir)
        print(f"\nProcessing {song_name}...")
        
        onsets_ms = extract_for_song(song_dir, model, chunk_sec=5.0)
        if not onsets_ms:
            continue

        out_dir = os.path.join(song_dir, "deepresonance_onsets")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = song_name.replace(" ", "_").replace("'", "")
        out_path = os.path.join(out_dir, f"{safe_name}_DeepResonance_5s.txt")

        with open(out_path, "w") as f:
            for ms in onsets_ms:
                f.write(f"{ms}\n")

        print(f"  ✅ Saved {len(onsets_ms)} onsets → {out_path}", flush=True)

if __name__ == "__main__":
    main()
