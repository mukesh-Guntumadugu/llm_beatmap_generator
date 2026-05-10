#!/usr/bin/env python3
import os
import sys
import csv
import re
import subprocess
import tempfile
from glob import glob

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUT_CSV = os.path.join(PROJ_DIR, "outputs", "zero_shot_deepresonance.csv")
PYTHON_BIN = "/data/mg546924/conda_envs/deepresonance_env/bin/python"

CODE_TEMPLATE = r'''
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
y = y[:int(sr*30)]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, sr)
    tmp_path = tmp.name

prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."

inputs = {
    "inputs": [prompt_bpm], "instructions": [prompt_bpm], "mm_names": [["audio"]],
    "mm_paths": [[os.path.basename(tmp_path)]], "mm_root_path": os.path.dirname(tmp_path), "outputs": [""],
}
resp_bpm = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
if isinstance(resp_bpm, list): resp_bpm = resp_bpm[0]
os.remove(tmp_path)

nums = re.findall(r"\d+\.?\d*", str(resp_bpm))
final_bpm = nums[0] if nums else "0.0"
print("BPM_RESPONSE=" + str(final_bpm))
'''

def run_model_query(audio_path):
    env = os.environ.copy()
    env["AUDIO_PATH"] = audio_path
    env["PROJ_DIR"] = PROJ_DIR
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(CODE_TEMPLATE)
        tmp = f.name
        
    try:
        r = subprocess.run([PYTHON_BIN, tmp], capture_output=True, text=True, timeout=600, env=env)
        out = r.stdout + r.stderr
        
        m_bpm = re.search(r'BPM_RESPONSE=(.*)', out)
        bpm_resp = m_bpm.group(1).strip() if m_bpm else "ERROR"
        
        if r.returncode != 0:
            print(f"    [!] Warning: STDERR tail = {r.stderr[-300:]}")
            
        return bpm_resp
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        os.remove(tmp)

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    songs = []
    for d in os.listdir(DATASET_DIR):
        full_dir = os.path.join(DATASET_DIR, d)
        if os.path.isdir(full_dir) and not d.startswith("_"):
            audio_files = glob.glob(os.path.join(full_dir, "*.ogg"))
            if audio_files:
                songs.append((d, audio_files[0]))
                
    songs = sorted(songs)[:20]
    print(f"Found {len(songs)} songs. Benchmarking DeepResonance...")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["song_name", "model", "pred_bpm"])
            
        for song_name, audio_path in songs:
            print(f"\n-> Querying DeepResonance for {song_name}...")
            pred_bpm = run_model_query(audio_path)
            print(f"   BPM: {pred_bpm}")
            writer.writerow([song_name, "DeepResonance", pred_bpm])
            f.flush()
            
if __name__ == "__main__":
    main()
