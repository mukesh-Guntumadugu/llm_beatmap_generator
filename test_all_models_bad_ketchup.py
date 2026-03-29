#!/usr/bin/env python3
"""
test_all_models_bad_ketchup.py — writes temp scripts to avoid quote issues
"""
import os
import sys
import csv
import re
import subprocess
import datetime
import tempfile

AUDIO_PATH = "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"
OUT_DIR    = "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup"
PROJ_DIR   = "/data/mg546924/llm_beatmap_generator"

PYTHON_DR  = "/data/mg546924/conda_envs/deepresonance_env/bin/python"
PYTHON_QWEN= "/data/mg546924/conda_envs/qwenenv/bin/python"
PYTHON_FL  = "/data/mg546924/music_flamingo_env/bin/python"

results = {}

def run_script(name, python_bin, code, timeout=1200):
    """Write code to a temp file and run it with python_bin."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run([python_bin, tmp], capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        m = re.search(r'ONSET_COUNT=(\d+)', out)
        count = int(m.group(1)) if m else "?"
        if r.returncode != 0 and not m:
            print(f"  STDERR tail: {r.stderr[-600:]}")
        return count
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR:{e}"
    finally:
        os.remove(tmp)


# ── 1. LIBROSA ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1/5  LIBROSA  (signal processing baseline)")
print("="*60)
try:
    import librosa
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onsets_ms = [round(float(t)*1000, 1) for t in onset_frames]
    ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    out_file = os.path.join(OUT_DIR, f"Librosa_TEST_Bad_Ketchup_{ts}.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["onset_ms"])
        for ms in onsets_ms: w.writerow([ms])
    results["Librosa"] = len(onsets_ms)
    print(f"✅ Librosa found {len(onsets_ms)} onsets → {out_file}")
except Exception as e:
    results["Librosa"] = f"ERROR:{e}"
    print(f"❌ Librosa failed: {e}")


# ── 2. DEEPRESONANCE ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2/5  DEEPRESONANCE  (LLaMA 7B + ImageBind)")
print("="*60)

dr_code = '''
import os, sys, gc, csv, datetime, re, tempfile, torch
import numpy as np

AUDIO = os.environ["BENCHMARK_AUDIO"]
OUT_DIR = os.environ["BENCHMARK_OUT"]
PROJ = os.environ["BENCHMARK_PROJ"]
CKPT = PROJ + "/DeepResonance/ckpt"

sys.path.insert(0, PROJ + "/DeepResonance/code")
os.chdir(PROJ + "/DeepResonance/code")

from inference_deepresonance import DeepResonancePredict
import librosa, soundfile as sf

args = {
    "stage": 2, "mode": "test", "dataset": "musiccaps",
    "project_path": PROJ + "/DeepResonance/code",
    "llm_path": CKPT + "/pretrained_ckpt/vicuna_ckpt/7b_v0",
    "imagebind_path": CKPT + "/pretrained_ckpt/imagebind_ckpt/huge",
    "imagebind_version": "huge",
    "max_length": 512, "max_output_length": 512,
    "num_clip_tokens": 77, "gen_emb_dim": 768,
    "preencoding_dropout": 0.1, "num_preencoding_layers": 1,
    "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False,
    "prompt": "", "prellmfusion": True, "prellmfusion_dropout": 0.1,
    "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.1,
    "ckpt_path": CKPT + "/DeepResonance_data_models/ckpt/deepresonance_beta_delta_ckpt/delta_ckpt/deepresonance/7b_tiva_v0",
}

y, sr = librosa.load(AUDIO, sr=None)
duration = len(y) / sr
CHUNK = 5
model = DeepResonancePredict(args)
all_onsets = []

for start in range(0, int(duration), CHUNK):
    end = min(start + CHUNK, duration)
    chunk = y[int(start*sr):int(end*sr)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, sr)
        tmp_path = tmp.name
    inputs = {
        "inputs": ["<Audio>"],
        "instructions": [f"List all onset timestamps in milliseconds for this {round(end-start,1)}s audio clip."],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]],
        "mm_root_path": os.path.dirname(tmp_path),
        "outputs": [""],
    }
    resp = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
    if isinstance(resp, list): resp = resp[0]
    nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", resp or "")
    for n in nums:
        all_onsets.append(round(float(n) + start*1000, 1))
    os.remove(tmp_path)
    gc.collect(); torch.cuda.empty_cache()

ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = OUT_DIR + "/DeepResonance_TEST_Bad_Ketchup_" + ts + ".csv"
with open(out, "w", newline="") as f:
    w = __import__("csv").writer(f)
    w.writerow(["onset_ms"])
    for ms in all_onsets: w.writerow([ms])

print(f"ONSET_COUNT={len(all_onsets)}")
print(f"OUTPUT_FILE={out}")
'''

env = os.environ.copy()
env["BENCHMARK_AUDIO"] = AUDIO_PATH
env["BENCHMARK_OUT"]   = OUT_DIR
env["BENCHMARK_PROJ"]  = PROJ_DIR

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(dr_code); tmp = f.name

try:
    r = subprocess.run([PYTHON_DR, tmp], capture_output=True, text=True, timeout=1200, env=env)
    out = r.stdout + r.stderr
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["DeepResonance"] = count
    print(f"✅ DeepResonance found {count} onsets")
    if r.returncode != 0: print(f"  STDERR: {r.stderr[-400:]}")
except Exception as e:
    results["DeepResonance"] = f"ERROR:{e}"
    print(f"❌ DeepResonance: {e}")
finally:
    os.remove(tmp)


# ── 3. QWEN ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3/5  QWEN AUDIO")
print("="*60)

qwen_code = '''
import os, sys, csv, datetime, re
AUDIO = os.environ["BENCHMARK_AUDIO"]
OUT_DIR = os.environ["BENCHMARK_OUT"]
PROJ = os.environ["BENCHMARK_PROJ"]
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")
os.chdir(PROJ)
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
setup_qwen()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_qwen(AUDIO, prompt)
nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", response or "")
onsets = [float(n) for n in nums]
ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = OUT_DIR + "/Qwen_TEST_Bad_Ketchup_" + ts + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in onsets: w.writerow([ms])
print(f"ONSET_COUNT={len(onsets)}")
print(f"OUTPUT_FILE={out}")
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(qwen_code); tmp = f.name
try:
    r = subprocess.run([PYTHON_QWEN, tmp], capture_output=True, text=True, timeout=600, env=env)
    out = r.stdout + r.stderr
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["Qwen"] = count
    print(f"✅ Qwen found {count} onsets")
    if r.returncode != 0: print(f"  STDERR: {r.stderr[-400:]}")
except Exception as e:
    results["Qwen"] = f"ERROR:{e}"
    print(f"❌ Qwen: {e}")
finally:
    os.remove(tmp)


# ── 4. MuMu-LLaMA ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4/5  MuMu-LLaMA")
print("="*60)

mumu_code = '''
import os, sys, csv, datetime, re
AUDIO = os.environ["BENCHMARK_AUDIO"]
OUT_DIR = os.environ["BENCHMARK_OUT"]
PROJ = os.environ["BENCHMARK_PROJ"]
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")
os.chdir(PROJ)
from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu
setup_mumu()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_mumu(AUDIO, prompt)
nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", response or "")
onsets = [float(n) for n in nums]
ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = OUT_DIR + "/MuMu_TEST_Bad_Ketchup_" + ts + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in onsets: w.writerow([ms])
print(f"ONSET_COUNT={len(onsets)}")
print(f"OUTPUT_FILE={out}")
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(mumu_code); tmp = f.name
try:
    r = subprocess.run([PYTHON_DR, tmp], capture_output=True, text=True, timeout=600, env=env)
    out = r.stdout + r.stderr
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["MuMu-LLaMA"] = count
    print(f"✅ MuMu-LLaMA found {count} onsets")
    if r.returncode != 0: print(f"  STDERR: {r.stderr[-400:]}")
except Exception as e:
    results["MuMu-LLaMA"] = f"ERROR:{e}"
    print(f"❌ MuMu-LLaMA: {e}")
finally:
    os.remove(tmp)


# ── 5. MUSIC-FLAMINGO ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5/5  MUSIC-FLAMINGO  (NVIDIA 30GB)")
print("="*60)

flamingo_code = '''
import os, sys, csv, datetime, re
AUDIO = os.environ["BENCHMARK_AUDIO"]
OUT_DIR = os.environ["BENCHMARK_OUT"]
PROJ = os.environ["BENCHMARK_PROJ"]
os.environ["HF_HOME"] = PROJ + "/Music-Flamingo/checkpoints"
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")
os.chdir(PROJ)
from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
setup_music_flamingo()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_flamingo(AUDIO, prompt)
nums = re.findall(r"\\b(\\d+(?:\\.\\d+)?)\\b", response or "")
onsets = [float(n) for n in nums]
ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = OUT_DIR + "/Flamingo_TEST_Bad_Ketchup_" + ts + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in onsets: w.writerow([ms])
print(f"ONSET_COUNT={len(onsets)}")
print(f"OUTPUT_FILE={out}")
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(flamingo_code); tmp = f.name
try:
    r = subprocess.run([PYTHON_FL, tmp], capture_output=True, text=True, timeout=600, env=env)
    out = r.stdout + r.stderr
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["Music-Flamingo"] = count
    print(f"✅ Music-Flamingo found {count} onsets")
    if r.returncode != 0: print(f"  STDERR: {r.stderr[-400:]}")
except Exception as e:
    results["Music-Flamingo"] = f"ERROR:{e}"
    print(f"❌ Music-Flamingo: {e}")
finally:
    os.remove(tmp)


# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK SUMMARY — Bad Ketchup onset detection")
print("="*60)
for model, count in results.items():
    status = "✅" if isinstance(count, int) else "❌"
    print(f"  {status}  {model:<20} →  {count} onsets")
print("="*60)
