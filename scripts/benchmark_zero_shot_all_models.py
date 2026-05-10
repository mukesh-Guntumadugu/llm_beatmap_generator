#!/usr/bin/env python3
"""
benchmark_zero_shot_all_models.py
=================================
Evaluates Zero-Shot performance for Onsets and BPM across 4 LLM models:
Qwen2-Audio, MuMu-LLaMA, DeepResonance, and Music-Flamingo.

Usage:
  python scripts/benchmark_zero_shot_all_models.py --max-songs 20
"""

import os
import sys
import csv
import re
import subprocess
import tempfile
import argparse
import datetime
from glob import glob
from tqdm import tqdm

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
BPM_JSON = os.path.join(PROJ_DIR, "local_bpm_analysis.json")
OUT_CSV = os.path.join(PROJ_DIR, "outputs", "zero_shot_results.csv")

# Paths to python binaries on the cluster
PYTHON_DR   = "/data/mg546924/conda_envs/deepresonance_env/bin/python"
PYTHON_QWEN = "/data/mg546924/conda_envs/qwenenv/bin/python"
PYTHON_FL   = "/data/mg546924/music_flamingo_env/bin/python"
# Mumu uses deepresonance_env

# --- GENERATED CODE TEMPLATES FOR EACH MODEL ---

CODE_TEMPLATE_DR = '''
import os, sys, gc, re, tempfile, torch
import librosa, soundfile as sf
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
CKPT = PROJ + "/DeepResonance/ckpt"
sys.path.insert(0, PROJ + "/DeepResonance/code")
os.chdir(PROJ + "/DeepResonance/code")
from inference_deepresonance import DeepResonancePredict

args = {
    "stage": 2, "mode": "test", "dataset": "musiccaps", "project_path": PROJ + "/DeepResonance/code",
    "llm_path": CKPT + "/pretrained_ckpt/vicuna_ckpt/7b_v0", "imagebind_path": CKPT + "/pretrained_ckpt/imagebind_ckpt/huge",
    "imagebind_version": "huge", "max_length": 512, "max_output_length": 512, "num_clip_tokens": 77, "gen_emb_dim": 768,
    "preencoding_dropout": 0.1, "num_preencoding_layers": 1, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False, "prompt": "", "prellmfusion": True,
    "prellmfusion_dropout": 0.1, "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.1,
    "ckpt_path": CKPT + "/DeepResonance_data_models/ckpt/deepresonance_beta_delta_ckpt/delta_ckpt/deepresonance/7b_tiva_v0",
}
model = DeepResonancePredict(args)
y, sr = librosa.load(AUDIO, sr=16000)
# Process max 30s to avoid OOM
y = y[:int(sr*30)]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, sr)
    tmp_path = tmp.name

prompt_onset = "List all the onset timestamps in this audio in milliseconds."
prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."

def ask(p):
    inputs = {
        "inputs": [p], "instructions": [p], "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]], "mm_root_path": os.path.dirname(tmp_path), "outputs": [""],
    }
    resp = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
    if isinstance(resp, list): resp = resp[0]
    return resp

resp_onset = ask(prompt_onset)
resp_bpm = ask(prompt_bpm)
os.remove(tmp_path)

nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", resp_onset or "")
print(f"ONSET_COUNT={len(nums)}")
print(f"BPM_RESPONSE={resp_bpm.strip().split('\\n')[0]}")
'''

CODE_TEMPLATE_QWEN = '''
import os, sys, re, librosa, soundfile as sf, tempfile
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
setup_qwen()
y, sr = librosa.load(AUDIO, sr=16000)
y = y[:int(sr*30)]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, sr)
    tmp_path = tmp.name

prompt_onset = "List all the onset timestamps in this audio in milliseconds."
prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."
resp_onset = generate_beatmap_with_qwen(tmp_path, prompt_onset)
resp_bpm = generate_beatmap_with_qwen(tmp_path, prompt_bpm)
os.remove(tmp_path)

nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", resp_onset or "")
print(f"ONSET_COUNT={len(nums)}")
print(f"BPM_RESPONSE={resp_bpm.strip().split('\\n')[0]}")
'''

CODE_TEMPLATE_MUMU = '''
import os, sys, re, librosa, soundfile as sf, tempfile
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu
setup_mumu()
y, sr = librosa.load(AUDIO, sr=16000)
y = y[:int(sr*30)]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, sr)
    tmp_path = tmp.name

prompt_onset = "List all the onset timestamps in this audio in milliseconds."
prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."
resp_onset = generate_beatmap_with_mumu(tmp_path, prompt_onset)
resp_bpm = generate_beatmap_with_mumu(tmp_path, prompt_bpm)
os.remove(tmp_path)

nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", resp_onset or "")
print(f"ONSET_COUNT={len(nums)}")
print(f"BPM_RESPONSE={resp_bpm.strip().split('\\n')[0]}")
'''

CODE_TEMPLATE_FLAMINGO = '''
import os, sys, re, librosa, soundfile as sf, tempfile
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
os.environ["HF_HOME"] = PROJ + "/Music-Flamingo/checkpoints"
sys.path.insert(0, PROJ)
from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
setup_music_flamingo()
y, sr = librosa.load(AUDIO, sr=16000)
y = y[:int(sr*30)]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, sr)
    tmp_path = tmp.name

prompt_onset = "List all the onset timestamps in this audio in milliseconds."
prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."
resp_onset = generate_beatmap_with_flamingo(tmp_path, prompt_onset)
resp_bpm = generate_beatmap_with_flamingo(tmp_path, prompt_bpm)
os.remove(tmp_path)

nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", resp_onset or "")
print(f"ONSET_COUNT={len(nums)}")
print(f"BPM_RESPONSE={resp_bpm.strip().split('\\n')[0]}")
'''

MODELS = {
    "Qwen":          (PYTHON_QWEN, CODE_TEMPLATE_QWEN),
    "MuMu":          (PYTHON_DR,   CODE_TEMPLATE_MUMU),
    "DeepResonance": (PYTHON_DR,   CODE_TEMPLATE_DR),
    "Flamingo":      (PYTHON_FL,   CODE_TEMPLATE_FLAMINGO)
}

def run_model_query(python_bin, code, audio_path):
    env = os.environ.copy()
    env["AUDIO_PATH"] = audio_path
    env["PROJ_DIR"] = PROJ_DIR
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp = f.name
        
    try:
        r = subprocess.run([python_bin, tmp], capture_output=True, text=True, timeout=600, env=env)
        out = r.stdout + r.stderr
        
        m_onset = re.search(r'ONSET_COUNT=(\\d+)', out)
        onset_count = int(m_onset.group(1)) if m_onset else 0
        
        m_bpm = re.search(r'BPM_RESPONSE=(.*)', out)
        bpm_resp = m_bpm.group(1).strip() if m_bpm else "ERROR"
        
        if r.returncode != 0 and onset_count == 0:
            print(f"    [!] Warning: STDERR tail = {r.stderr[-300:]}")
            
        return onset_count, bpm_resp
    except subprocess.TimeoutExpired:
        return 0, "TIMEOUT"
    except Exception as e:
        return 0, f"ERROR: {e}"
    finally:
        os.remove(tmp)

def get_ground_truth(song_dir):
    gt_onsets = 0
    csvs = glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if csvs:
        csv_path = max(csvs, key=os.path.getmtime)
        try:
            with open(csv_path, "r") as f:
                gt_onsets = sum(1 for _ in f) - 1 # skip header
        except: pass
    return gt_onsets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-songs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    
    songs = []
    for d in os.listdir(DATASET_DIR):
        full_dir = os.path.join(DATASET_DIR, d)
        if os.path.isdir(full_dir) and not d.startswith("_"):
            audio_files = glob(os.path.join(full_dir, "*.ogg"))
            if audio_files:
                songs.append((d, audio_files[0]))
                
    songs = sorted(songs)[:args.max_songs]
    print(f"Found {len(songs)} songs to evaluate.")

    file_exists = os.path.isfile(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["song_name", "model", "gt_onsets", "pred_onsets", "pred_bpm"])
            
        for song_name, audio_path in tqdm(songs, desc="Evaluating Songs"):
            gt_onsets = get_ground_truth(os.path.dirname(audio_path))
            
            for model_name, (py_bin, code) in MODELS.items():
                print(f"\\n  -> Querying {model_name} for {song_name}...")
                pred_onsets, pred_bpm = run_model_query(py_bin, code, audio_path)
                
                print(f"     Onsets: {pred_onsets} (GT: {gt_onsets}) | BPM: {pred_bpm}")
                writer.writerow([song_name, model_name, gt_onsets, pred_onsets, pred_bpm])
                f.flush()

    print(f"\\n✅ Zero-Shot Benchmarking complete! Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
