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
OUT_CSV = os.path.join(PROJ_DIR, "outputs", "zero_shot_flamingo.csv")
PYTHON_BIN = "/data/mg546924/conda_envs/flamingo_env/bin/python"

CODE_TEMPLATE = r'''
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

prompt_bpm = "What is the exact BPM and tempo of this song? Only output the number."
resp_bpm = generate_beatmap_with_flamingo(tmp_path, prompt_bpm)
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
    print(f"Found {len(songs)} songs. Benchmarking Flamingo...")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["song_name", "model", "pred_bpm"])
            
        for song_name, audio_path in songs:
            print(f"\n-> Querying Flamingo for {song_name}...")
            pred_bpm = run_model_query(audio_path)
            print(f"   BPM: {pred_bpm}")
            writer.writerow([song_name, "Flamingo", pred_bpm])
            f.flush()
            
if __name__ == "__main__":
    main()
