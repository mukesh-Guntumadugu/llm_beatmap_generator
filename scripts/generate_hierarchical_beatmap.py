#!/usr/bin/env python3
"""
generate_hierarchical_beatmap.py
================================
Phase 4 of the Hierarchical Architecture.

This is the Runtime Inference Engine:
1. Slices a target audio file into 5-second contiguous chunks.
2. The fine-tuned MuMu Director LLM predicts a grammar sequence of <|cluster_N|> tokens.
3. The Actor decrypts these tokens using the pattern dictionary and stamps
   real, 192-tick physical measures.
4. Outputs a final, formatted .ssc beatmap file.

Usage:
  python generate_hierarchical_beatmap.py --audio /path/to/song.wav --bpm 130
"""

import os
import sys
import json
import torch
import argparse
import random
import torchaudio
import numpy as np
import uuid

# ── Paths ──
MUMU_ROOT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/MuMu-LLaMA"
sys.path.insert(0, MUMU_ROOT)

DICT_PATH   = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns.json"
MODELS_DIR  = "/data/mg546924/models/mumu-hierarchical-director"

# Hyperparameters
MAX_WORDS   = 256
SAMPLE_RATE = 24000
CHUNK_SEC   = 5.0

def load_actor_dictionary(dict_path):
    print(f"Loading Actor Sub-Decoder dictionary from {dict_path}...")
    with open(dict_path, "r") as f:
        cluster_dict = json.load(f)
    print(f"  Loaded {len(cluster_dict)} physical cluster mappings.")
    return cluster_dict

def extract_tokens_from_response(text):
    """Extracts <|cluster_X|> from raw text."""
    import re
    tokens = re.findall(r"<\|cluster_\d+\|>", text)
    return tokens

def generate_beatmap(audio_path, out_ssc_path, bpm, difficulty="Challenge"):
    os.makedirs(os.path.dirname(out_ssc_path) if os.path.dirname(out_ssc_path) else ".", exist_ok=True)
    
    # 1. Load Actor Map
    cluster_dict = load_actor_dictionary(DICT_PATH)
    
    # 2. Setup MuMu Model
    from llama.mumu_llama import MuMu_LLaMA
    from transformers import LlamaTokenizer
    import llama
    
    print("Loading specialized cluster Tokenizer...")
    # Load the extended tokenizer saved during training!
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    
    model_args = argparse.Namespace(
        mert_path="m-a-p/MERT-v1-330M",
        vit_path="google/vit-base-patch16-224",
        vivit_path="google/vivit-b-16x2-kinetics400",
        music_decoder="musicgen",
        music_decoder_path="facebook/musicgen-small",
        max_words=MAX_WORDS,
    )
    
    print("Loading MuMu Director LLM...")
    model = MuMu_LLaMA(
        llama_ckpt_dir="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA/7B",
        llama_tokenizer="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA",
        model_args=model_args,
        knn_dir="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts",
        stage=3,
    )
    
    # Resize the embeddings to match our new extended tokenizer BEFORE loading weights!
    model.llama.resize_token_embeddings(len(tokenizer))
    
    # Find the latest checkpoint
    import glob
    ckpts = glob.glob(os.path.join(MODELS_DIR, "checkpoint_epoch*.pth"))
    if not ckpts:
        print("ERROR: No checkpoints found. Is training done?")
        sys.exit(1)
    # Sort and pick highest epoch
    latest_ckpt = sorted(ckpts)[-1]
    
    print(f"Loading weights from {latest_ckpt}...")
    state_dict = torch.load(latest_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().bfloat16().eval()
    
    # 3. Process Audio
    print(f"\nAnalyzing Audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    audio_full = torch.mean(waveform, 0) # Mono
    
    duration = audio_full.shape[0] / SAMPLE_RATE
    print(f"Total Duration: {duration:.2f}s")
    
    all_cluster_tokens = []
    
    # 4. Sliding Window Inference
    print("\nRunning Director Inference...")
    for win_start in np.arange(0, duration, CHUNK_SEC):
        win_end = win_start + CHUNK_SEC
        if win_end > duration:
            # Skip the trailing partial window if it's too small
            if (duration - win_start) < (CHUNK_SEC * 0.5):
                break
            win_end = duration
            
        start_idx = int(win_start * SAMPLE_RATE)
        end_idx   = int(win_end * SAMPLE_RATE)
        audio_chunk = audio_full[start_idx:end_idx]
        
        # Pad if short
        if audio_chunk.shape[0] < int(CHUNK_SEC * SAMPLE_RATE):
            pad = int(CHUNK_SEC * SAMPLE_RATE) - audio_chunk.shape[0]
            audio_chunk = torch.cat([audio_chunk, torch.zeros(pad)])
            
        # Director Prompt
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this {CHUNK_SEC}s audio segment. "
            f"The difficulty is {difficulty}. "
            "Predict the ordered sequence of rhythmic pattern cluster tokens."
        )
        input_text = llama.utils.format_prompt(prompt)
        input_ids  = tokenizer(input_text).input_ids
        input_ids  = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).cuda()
        audio_t    = audio_chunk.cuda().float().unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    audios=audio_t,
                    max_new_tokens=20,
                    do_sample=True,    # A bit of temperature for creative variety
                    temperature=0.7,
                    top_p=0.9
                )
                
        # Decode prediction
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        tokens = extract_tokens_from_response(response)
        
        print(f"  [{win_start:04.1f}s - {win_end:04.1f}s]: {' '.join(tokens)}")
        all_cluster_tokens.extend(tokens)
        
    # 5. Actor Translation
    print("\nRunning Actor Translation to Physical Measures...")
    physical_measures = []
    for t in all_cluster_tokens:
        if t in cluster_dict and cluster_dict[t]:
            # Pick a dynamic physical pattern variation for this topology cluster
            pattern_str = random.choice(cluster_dict[t])
            physical_measures.append(pattern_str)
        else:
            # Fallback to empty measure if token is hallucinated/missing
            physical_measures.append("\n".join(["0000" for _ in range(192)]))
            
    # 6. SSC Formatting
    print("\nFormatting SSC File...")
    ssc_header = f"""#VERSION:0.83;
#TITLE:Generated Hierarchical Chart;
#SUBTITLE:;
#ARTIST:MuMu Director x Actor SubDecoder;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#ORIGIN:;
#CREDIT:AI;
#BANNER:;
#BACKGROUND:;
#PREVIEWVID:;
#JACKET:;
#CDIMAGE:;
#DISCIMAGE:;
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:{os.path.basename(audio_path)};
#OFFSET:0.000000;
#SAMPLESTART:0.000000;
#SAMPLELENGTH:10.000000;
#SELECTABLE:YES;
#BPMS:0.000000={bpm:.6f};
#STOPS:;
#DELAYS:;
#WARPS:;
#TIMESIGNATURES:0.000000=4=4;
#TICKCOUNTS:0.000000=4;
#COMBOS:0.000000=1;
#SPEEDS:0.000000=1.000000=0.000000=0;
#SCROLLS:0.000000=1.000000;
#FAKES:;
#LABELS:0.000000=Song Start;
#BGCHANGES:;
#KEYSOUNDS:;
#ATTACKS:;"""

    measures_str = ",\n".join(physical_measures)
    ssc_chart = f"""
//---------------dance-single - ----------------
#NOTEDATA:;
#CHARTNAME:;
#STEPSTYPE:dance-single;
#DESCRIPTION:;
#CHARTSTYLE:;
#DIFFICULTY:{difficulty};
#METER:9;
#RADARVALUES:0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000;
#CREDIT:;
#NOTES:
{measures_str}
;"""

    with open(out_ssc_path, "w", encoding="utf-8") as f:
        f.write(ssc_header + ssc_chart)
        
    print(f"\n==========================================")
    print(f"[OK] Hierarchical Beatmap Generated!")
    print(f"  Input Music   : {audio_path}")
    print(f"  Chart Output  : {out_ssc_path}")
    print(f"  Total Tokens  : {len(all_cluster_tokens)}")
    print(f"  Total Measures: {len(physical_measures)}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--bpm", type=float, default=130.0, help="Uniform BPM to set for the SSC file")
    parser.add_argument("--difficulty", default="Challenge", help="Target difficulty")
    parser.add_argument("--out", default="output.ssc", help="Path to output SSC file")
    args = parser.parse_args()
    
    generate_beatmap(args.audio, args.out, args.bpm, args.difficulty)
