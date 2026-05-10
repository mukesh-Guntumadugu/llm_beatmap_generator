#!/usr/bin/env python3
"""
benchmark_sft_onsets_direct.py
==============================
Directly evaluates the precise SFT millisecond generation for 
all 4 models across the 20 test songs.

Bypasses StepMania formatting to prevent grid snapping and
preserves raw LLM output accuracy.
"""

import os
import sys
import csv
import re
import subprocess
import tempfile
import argparse
import numpy as np
import librosa
from glob import glob
from tqdm import tqdm

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUT_CSV = os.path.join(PROJ_DIR, "outputs", "sft_direct_onset_results.csv")
TOLERANCE_MS = 50.0

# Paths to python binaries on the cluster
PYTHON_DR   = "/data/mg546924/conda_envs/deepresonance_env/bin/python"
PYTHON_QWEN = "/data/mg546924/conda_envs/qwenenv/bin/python"
PYTHON_FL   = "/data/mg546924/music_flamingo_env/bin/python"

# ─────────────────────────────────────────────
# Subprocess Templates (LLM SFT INFERENCE)
# ─────────────────────────────────────────────

CODE_TEMPLATE_QWEN = r'''
import os, sys, re, librosa, torch, soundfile as sf, tempfile, numpy as np
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR   = "/data/mg546924/models/qwen2-audio-lora-onsets" # Use the original onset LoRA, not the hierarchical one
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = Qwen2AudioForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
base_model.resize_token_embeddings(len(processor.tokenizer))
try:
    # Try using checkpoint-2208 if it exists, otherwise root
    if os.path.exists(LORA_DIR + "/checkpoint-2208"):
        model = PeftModel.from_pretrained(base_model, LORA_DIR + "/checkpoint-2208").eval()
    else:
        model = PeftModel.from_pretrained(base_model, LORA_DIR).eval()
except:
    model = base_model.eval()

y_full, sr = librosa.load(AUDIO, sr=processor.feature_extractor.sampling_rate)
duration = librosa.get_duration(y=y_full, sr=sr)
CHUNK_SEC = 5.0

all_onsets = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    if win_end - win_start < 4.9: continue
    
    y_chunk = y_full[int(win_start*sr):int(win_end*sr)]
    prompt = f"You are a rhythm game beatmap generator. This is a {round(win_end - win_start, 1)}s audio clip. List ONLY the onset timestamps in milliseconds as plain numbers separated by commas."
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=text, audio=[y_chunk], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    resp = processor.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    nums = re.findall(r"\d+\.?\d*", resp)
    for n in nums:
        all_onsets.append(str(int(round(float(n) + win_start*1000))))

print("ONSETS_RESPONSE=" + ",".join(all_onsets))
'''

CODE_TEMPLATE_MUMU = r'''
import os, sys, re, librosa, torch, soundfile as sf, tempfile, numpy as np
import argparse
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/MuMu-LLaMA/MuMu-LLaMA")
from llama.mumu_llama import MuMu_LLaMA
from transformers import LlamaTokenizer
import llama

MODELS_DIR = "/data/mg546924/models/mumu-llama-lora-onsets"
tokenizer = LlamaTokenizer.from_pretrained(PROJ+"/MuMu-LLaMA/ckpts/LLaMA")

model_args = argparse.Namespace(mert_path="m-a-p/MERT-v1-330M", vit_path="google/vit-base-patch16-224", vivit_path="google/vivit-b-16x2-kinetics400", music_decoder="musicgen", music_decoder_path="facebook/musicgen-medium", max_words=512)
model = MuMu_LLaMA(llama_ckpt_dir=PROJ+"/MuMu-LLaMA/ckpts/LLaMA/7B", llama_tokenizer=PROJ+"/MuMu-LLaMA/ckpts/LLaMA", model_args=model_args, knn_dir=PROJ+"/MuMu-LLaMA/ckpts", stage=3)

import glob
try:
    latest_ckpt = sorted(glob.glob(MODELS_DIR + "/checkpoint_epoch*.pth"))[-1]
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
except:
    pass
model = model.cuda().float().eval()

import torchaudio
waveform, sr = torchaudio.load(AUDIO)
if sr != 24000: waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
audio_mono = torch.mean(waveform, 0)
duration = audio_mono.shape[0] / 24000
CHUNK_SEC = 5.0

all_onsets = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    if win_end - win_start < 4.9: continue
    
    chunk = audio_mono[int(win_start*24000):int(win_end*24000)]
    if chunk.shape[0] < int(CHUNK_SEC * 24000):
        chunk = torch.cat([chunk, torch.zeros(int(CHUNK_SEC * 24000) - chunk.shape[0])])
    
    prompt = f"You are a rhythm game beatmap generator. This is a 5.0s audio clip. List ONLY the onset timestamps in milliseconds as plain numbers separated by commas."
    input_ids = torch.tensor(tokenizer(llama.utils.format_prompt(prompt)).input_ids, dtype=torch.int64).unsqueeze(0).cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        output_ids = model.generate(input_ids, audios=chunk.cuda().float(), max_gen_len=100)
    resp = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    nums = re.findall(r"\d+\.?\d*", resp)
    for n in nums:
        all_onsets.append(str(int(round(float(n) + win_start*1000))))

print("ONSETS_RESPONSE=" + ",".join(all_onsets))
'''

CODE_TEMPLATE_DR = r'''
import os, sys, gc, re, tempfile, torch, numpy as np
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
    "ckpt_path": "/data/mg546924/models/deepresonance-lora-onsets",
}
try:
    model = DeepResonancePredict(args)
except:
    args["ckpt_path"] = CKPT + "/DeepResonance_data_models/ckpt/deepresonance_beta_delta_ckpt/delta_ckpt/deepresonance/7b_tiva_v0"
    model = DeepResonancePredict(args)

y_full, sr = librosa.load(AUDIO, sr=16000)
duration = librosa.get_duration(y=y_full, sr=sr)
CHUNK_SEC = 5.0

all_onsets = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    if win_end - win_start < 4.9: continue
    
    y_chunk = y_full[int(win_start*sr):int(win_end*sr)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y_chunk, sr)
        tmp_path = tmp.name
        
    prompt = f"You are a rhythm game beatmap generator. This is a {round(win_end - win_start, 1)}s audio clip. List ONLY the onset timestamps in milliseconds as plain numbers separated by commas."
    inputs = {
        "inputs": [prompt], "instructions": [prompt], "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]], "mm_root_path": os.path.dirname(tmp_path), "outputs": [""],
    }
    resp = model.predict(inputs, max_tgt_len=100, top_p=0.9, temperature=0.7, stops_id=[[835]])
    if isinstance(resp, list): resp = resp[0]
    os.remove(tmp_path)
    
    nums = re.findall(r"\d+\.?\d*", str(resp))
    for n in nums:
        all_onsets.append(str(int(round(float(n) + win_start*1000))))

print("ONSETS_RESPONSE=" + ",".join(all_onsets))
'''

CODE_TEMPLATE_FLAMINGO = r'''
import os, sys, re, librosa, soundfile as sf, tempfile, numpy as np
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
os.environ["HF_HOME"] = PROJ + "/Music-Flamingo/checkpoints"
sys.path.insert(0, PROJ)
from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
setup_music_flamingo()

y_full, sr = librosa.load(AUDIO, sr=16000)
duration = librosa.get_duration(y=y_full, sr=sr)
CHUNK_SEC = 5.0

all_onsets = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    if win_end - win_start < 4.9: continue
    
    y_chunk = y_full[int(win_start*sr):int(win_end*sr)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y_chunk, sr)
        tmp_path = tmp.name
        
    prompt = f"You are a rhythm game beatmap generator. This is a {round(win_end - win_start, 1)}s audio clip. List ONLY the onset timestamps in milliseconds as plain numbers separated by commas."
    resp = generate_beatmap_with_flamingo(tmp_path, prompt)
    os.remove(tmp_path)
    
    nums = re.findall(r"\d+\.?\d*", str(resp))
    for n in nums:
        all_onsets.append(str(int(round(float(n) + win_start*1000))))

print("ONSETS_RESPONSE=" + ",".join(all_onsets))
'''

MODELS = {
    "Qwen":          (PYTHON_QWEN, CODE_TEMPLATE_QWEN),
    "MuMu":          (PYTHON_DR,   CODE_TEMPLATE_MUMU),
    "DeepResonance": (PYTHON_DR,   CODE_TEMPLATE_DR),
    "Flamingo":      (PYTHON_FL,   CODE_TEMPLATE_FLAMINGO)
}

# ─────────────────────────────────────────────
# Scoring Logic
# ─────────────────────────────────────────────

def load_ground_truth(song_name):
    song_dir = os.path.join(DATASET_DIR, song_name)
    csvs = glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if not csvs: return []
    csv_path = max(csvs, key=os.path.getmtime)
    
    onsets_ms = []
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
        for line in lines[1:]: 
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try: onsets_ms.append(float(parts[1]))
                except ValueError: pass
    except: pass
    return sorted(onsets_ms)

def score_onsets(predictions_ms: list[float], ground_truth_ms: list[float]):
    if not predictions_ms or not ground_truth_ms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predictions_ms), "fn": len(ground_truth_ms)}

    pred_arr = np.array(predictions_ms)
    gt_arr   = np.array(ground_truth_ms)
    matched_gt = set()
    tp = 0

    for p in pred_arr:
        diffs = np.abs(gt_arr - p)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= TOLERANCE_MS and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_arr) - tp
    fn = len(gt_arr)   - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

def run_model(python_bin, code, audio_path):
    env = os.environ.copy()
    env["AUDIO_PATH"] = audio_path
    env["PROJ_DIR"] = PROJ_DIR
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp = f.name
        
    try:
        r = subprocess.run([python_bin, tmp], capture_output=True, text=True, timeout=1800, env=env)
        out = r.stdout + r.stderr
        m_tokens = re.search(r'ONSETS_RESPONSE=(.*)', out)
        if m_tokens:
            token_str = m_tokens.group(1).strip()
            return [float(t) for t in token_str.split(",") if t]
        print(f"    [!] Error running director: {r.stderr[-300:]}")
        return []
    finally:
        os.remove(tmp)

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
            if audio_files: songs.append((d, audio_files[0]))
                
    songs = sorted(songs)[:args.max_songs]
    print(f"Found {len(songs)} songs for Direct SFT Onset Evaluation.")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["song_name", "model", "gt_onsets", "pred_onsets", "precision", "recall", "f1", "tp", "fp", "fn"])
        
        for song_name, audio_path in tqdm(songs, desc="Benchmarking Direct Onsets"):
            gt_onsets_ms = load_ground_truth(song_name)
            if not gt_onsets_ms: continue
            
            for model_name, (py_bin, code) in MODELS.items():
                print(f"\n  -> Querying {model_name} for {song_name}...")
                pred_onsets_ms = run_model(py_bin, code, audio_path)
                
                scores = score_onsets(pred_onsets_ms, gt_onsets_ms)
                print(f"     ✅ {model_name} | GT: {len(gt_onsets_ms)} | Pred: {len(pred_onsets_ms)} | P: {scores['precision']:.3f} | R: {scores['recall']:.3f} | F1: {scores['f1']:.3f}")
                
                writer.writerow([
                    song_name, model_name, len(gt_onsets_ms), len(pred_onsets_ms),
                    scores["precision"], scores["recall"], scores["f1"],
                    scores["tp"], scores["fp"], scores["fn"]
                ])
                f.flush()

    print(f"\n✅ Direct SFT Onset Benchmarking complete! Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
