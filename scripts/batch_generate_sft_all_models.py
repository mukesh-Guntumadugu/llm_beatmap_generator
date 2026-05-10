#!/usr/bin/env python3
"""
batch_generate_sft_all_models.py
================================
Batch generator for the SFT evaluation on 20 test songs.
Separates the Director (subprocess LLM inference) from the 
Actor (deterministic token-to-measure mapping).
"""

import os
import sys
import json
import re
import subprocess
import tempfile
import argparse
import numpy as np
import librosa
from glob import glob
# pyrefly: ignore [missing-source-for-stubs]
from tqdm import tqdm

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
BPM_JSON = os.path.join(PROJ_DIR, "local_bpm_analysis.json")
OUT_DIR = os.path.join(PROJ_DIR, "outputs")
DICT_PATH = os.path.join(PROJ_DIR, "scripts", "cluster_to_patterns.json")

# Paths to python binaries on the cluster
PYTHON_DR   = "/data/mg546924/conda_envs/deepresonance_env/bin/python"
PYTHON_QWEN = "/data/mg546924/conda_envs/qwenenv/bin/python"
PYTHON_FL   = "/data/mg546924/music_flamingo_env/bin/python"

# ─────────────────────────────────────────────
# Subprocess Templates (LLM DIRECTOR ONLY)
# These only extract tokens for 5s chunks.
# ─────────────────────────────────────────────

CODE_TEMPLATE_QWEN = r'''
import os, sys, re, librosa, torch, soundfile as sf, tempfile, numpy as np
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
BPM = float(os.environ["BPM"])
sys.path.insert(0, PROJ)
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR   = "/data/mg546924/models/qwen2-audio-hierarchical-director"
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
with open(PROJ + "/scripts/cluster_to_patterns_tokens.txt", "r") as f:
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [t.strip() for t in f if t.strip()]})

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = Qwen2AudioForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
base_model.resize_token_embeddings(len(processor.tokenizer))
try:
    model = PeftModel.from_pretrained(base_model, LORA_DIR).eval()
except:
    model = base_model.eval()

y_full, sr = librosa.load(AUDIO, sr=processor.feature_extractor.sampling_rate)
duration = librosa.get_duration(y=y_full, sr=sr)
CHUNK_SEC = 5.0

all_tokens = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    y_chunk = y_full[int(win_start*sr):int(win_end*sr)]
    prompt = f"Listen to this {round(win_end - win_start, 1)}s audio segment. Song BPM: {BPM:.1f}. Predict the ordered sequence of rhythmic pattern cluster tokens."
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=text, audio=[y_chunk], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    resp = processor.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    tokens = re.findall(r"<\|cluster_\d+\|>", resp)
    all_tokens.extend(tokens)

print("TOKENS_RESPONSE=" + ",".join(all_tokens))
'''

CODE_TEMPLATE_MUMU = r'''
import os, sys, re, librosa, torch, soundfile as sf, tempfile, numpy as np
import argparse
AUDIO = os.environ["AUDIO_PATH"]
PROJ = os.environ["PROJ_DIR"]
BPM = float(os.environ["BPM"])
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/MuMu-LLaMA/MuMu-LLaMA")
from llama.mumu_llama import MuMu_LLaMA
from transformers import LlamaTokenizer
import llama

MODELS_DIR = "/data/mg546924/models/mumu-hierarchical-director"
tokenizer = LlamaTokenizer.from_pretrained(MODELS_DIR + "/tokenizer")

with open(PROJ + "/scripts/cluster_to_patterns_tokens.txt", "r") as f:
    tokenizer.add_special_tokens({'additional_special_tokens': [t.strip() for t in f if t.strip()]})

model_args = argparse.Namespace(mert_path="m-a-p/MERT-v1-330M", vit_path="google/vit-base-patch16-224", vivit_path="google/vivit-b-16x2-kinetics400", music_decoder="musicgen", music_decoder_path="facebook/musicgen-small", max_words=256)
model = MuMu_LLaMA(llama_ckpt_dir=PROJ+"/MuMu-LLaMA/ckpts/LLaMA/7B", llama_tokenizer=PROJ+"/MuMu-LLaMA/ckpts/LLaMA", model_args=model_args, knn_dir=PROJ+"/MuMu-LLaMA/ckpts", stage=3)

import torch.nn as nn
new_vocab = len(tokenizer)
old_emb = model.llama.tok_embeddings
old_size, dim = old_emb.weight.shape
if new_vocab != old_size:
    new_emb = nn.Embedding(new_vocab, dim)
    new_emb.weight.data[:old_size] = old_emb.weight.data
    model.llama.tok_embeddings = new_emb
    if hasattr(model.llama, 'output'):
        new_out = nn.Linear(dim, new_vocab, bias=False)
        new_out.weight.data[:old_size] = model.llama.output.weight.data
        model.llama.output = new_out

import glob
latest_ckpt = sorted(glob.glob(MODELS_DIR + "/checkpoint_epoch*.pth"))[-1]
model.load_state_dict(torch.load(latest_ckpt, map_location="cpu"), strict=True)
model = model.cuda().bfloat16().eval()

import torchaudio
waveform, sr = torchaudio.load(AUDIO)
if sr != 24000: waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
audio_mono = torch.mean(waveform, 0)
duration = audio_mono.shape[0] / 24000
CHUNK_SEC = 5.0

all_tokens = []
for win_start in np.arange(0, duration, CHUNK_SEC):
    win_end = min(win_start + CHUNK_SEC, duration)
    chunk = audio_mono[int(win_start*24000):int(win_end*24000)]
    if chunk.shape[0] < int(CHUNK_SEC * 24000):
        chunk = torch.cat([chunk, torch.zeros(int(CHUNK_SEC * 24000) - chunk.shape[0])])
    
    prompt = f"You are a rhythm game beatmap generator. Listen to this 5-second audio segment. The song BPM is {BPM:.1f}. Difficulty: Challenge. Predict the ordered sequence of rhythmic pattern cluster tokens."
    input_ids = torch.tensor(tokenizer(llama.utils.format_prompt(prompt)).input_ids, dtype=torch.int64).unsqueeze(0).cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = model.generate(input_ids, audios=chunk.cuda().float(), max_gen_len=20)
    resp = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    tokens = re.findall(r"<\|cluster_\d+\|>", resp)
    all_tokens.extend(tokens)

print("TOKENS_RESPONSE=" + ",".join(all_tokens))
'''

# Currently, we only implement Qwen and MuMu because DR and Flamingo lack mature hierarchical LoRAs.
# The orchestrator will only process these two to avoid crashing the pipeline.

MODELS = {
    "Qwen": (PYTHON_QWEN, CODE_TEMPLATE_QWEN),
    "MuMu": (PYTHON_QWEN, CODE_TEMPLATE_MUMU)
}

# ─────────────────────────────────────────────
# Actor Logic (Deterministic mapping)
# ─────────────────────────────────────────────

def align_tokens_to_measures(tokens, target_len):
    if not tokens:
        return ["<|cluster_0|>"] * target_len
    aligned = []
    for i in range(target_len):
        idx = int(i * len(tokens) / target_len)
        aligned.append(tokens[idx])
    return aligned

def tokens_to_measures(tokens, cluster_dict, difficulty_name="Challenge"):
    bias = {"Beginner": 0, "Easy": 1, "Medium": 2, "Hard": 3, "Challenge": 4}.get(difficulty_name, 4)
    measures = []
    for t in tokens:
        if t in cluster_dict and cluster_dict[t]:
            patterns = cluster_dict[t]
            sorted_patterns = sorted(patterns, key=lambda p: sum(1 for r in p.split("\n") if any(c != "0" for c in r)))
            idx = min(bias, len(sorted_patterns) - 1)
            measures.append(sorted_patterns[idx])
        else:
            measures.append("\n".join(["0000"] * 16))
    return measures

def generate_ssc(song_title, bpm, measures, out_path):
    measures_str = ",\n".join(measures)
    ssc = f"""#VERSION:0.83;
#TITLE:{song_title};
#MUSIC:{song_title}.ogg;
#BPMS:0.000000={bpm:.6f};
#OFFSET:0.000000;
//---------------dance-single - Challenge----------------
#NOTEDATA:;
#CHARTNAME:Challenge;
#STEPSTYPE:dance-single;
#DIFFICULTY:Challenge;
#METER:9;
#NOTES:
{measures_str}
;"""
    with open(out_path, "w") as f:
        f.write(ssc)

# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

def run_director(python_bin, code, audio_path, bpm):
    env = os.environ.copy()
    env["AUDIO_PATH"] = audio_path
    env["PROJ_DIR"] = PROJ_DIR
    env["BPM"] = str(bpm)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp = f.name
        
    try:
        r = subprocess.run([python_bin, tmp], capture_output=True, text=True, timeout=1200, env=env)
        out = r.stdout + r.stderr
        
        m_tokens = re.search(r'TOKENS_RESPONSE=(.*)', out)
        if m_tokens:
            token_str = m_tokens.group(1).strip()
            return [t for t in token_str.split(",") if t]
        
        print(f"    [!] Error running director: {r.stderr[-300:]}")
        return []
    finally:
        os.remove(tmp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-songs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DICT_PATH, "r") as f:
        cluster_dict = json.load(f)
        
    with open(BPM_JSON, "r") as f:
        bpm_db = json.load(f)
        
    songs = []
    for d in os.listdir(DATASET_DIR):
        full_dir = os.path.join(DATASET_DIR, d)
        if os.path.isdir(full_dir) and not d.startswith("_"):
            audio_files = glob(os.path.join(full_dir, "*.ogg"))
            if audio_files:
                # Find BPM
                gt_bpm = 130.0
                for sm_key, sm_data in bpm_db.items():
                    if sm_key.replace(".sm", "").replace(".ssc", "") == d:
                        if "sections" in sm_data and len(sm_data["sections"]) > 0:
                            gt_bpm = sm_data["sections"][0]["bpm"]
                        break
                songs.append((d, audio_files[0], gt_bpm))
                
    songs = sorted(songs)[:args.max_songs]
    print(f"Found {len(songs)} songs to evaluate SFT.")

    for song_name, audio_path, bpm in tqdm(songs, desc="Batch Generating"):
        duration = librosa.get_duration(filename=audio_path)
        total_beats = duration * (bpm / 60.0)
        target_measures = int(np.round(total_beats / 4.0))
        
        for model_name, (py_bin, code) in MODELS.items():
            out_file = os.path.join(OUT_DIR, f"{model_name.lower()}_{song_name.replace(' ', '_')}.ssc")
            if os.path.exists(out_file):
                continue
                
            print(f"\n  -> Generating {model_name} for {song_name} ({bpm} BPM)...")
            tokens = run_director(py_bin, code, audio_path, bpm)
            
            if tokens:
                aligned = align_tokens_to_measures(tokens, target_measures)
                measures = tokens_to_measures(aligned, cluster_dict, "Challenge")
                generate_ssc(song_name, bpm, measures, out_file)
                print(f"     ✅ Saved {len(measures)} measures to {os.path.basename(out_file)}")
            else:
                print(f"     ❌ Failed to generate tokens.")

    print(f"\n✅ Batch Generation complete!")

if __name__ == "__main__":
    main()
