#!/usr/bin/env python3
"""
generate_hierarchical_beatmap_qwen.py
=====================================
Phase 4 of the Hierarchical Architecture using Qwen2-Audio.

This is the Runtime Inference Engine:
1. Slices a target audio file into 5-second contiguous chunks.
2. The fine-tuned Qwen Director LLM predicts a sequence of <|cluster_N|> tokens.
3. The Actor decrypts these tokens using the pattern dictionary and stamps
   real, 192-tick physical measures.
4. Outputs a final, formatted .ssc beatmap file.

Usage:
  python generate_hierarchical_beatmap_qwen.py --audio /path/to/song.wav --bpm 130
"""

import os
import sys
import json
import torch
import argparse
import random
import torchaudio
import numpy as np
import librosa

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# ── Paths ──
DICT_PATH   = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns.json"
BASE_MODEL  = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR    = "/data/mg546924/models/qwen2-audio-hierarchical-director"

# Hyperparameters
SAMPLE_RATE = 16000 # Qwen default for audio processor
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
    
    # 2. Setup Qwen Model
    print(f"Loading Qwen Processor from {LORA_DIR}...")
    try:
        processor = AutoProcessor.from_pretrained(LORA_DIR, trust_remote_code=True)
    except Exception:
        print("Fallback to base processor...")
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        # We must load tokens manually if fallback
        TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
        with open(TOKENS_TXT, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    print("Loading Base Model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print(f"Resizing embeddings to {len(processor.tokenizer)}...")
    base_model.resize_token_embeddings(len(processor.tokenizer))
    
    if os.path.exists(LORA_DIR):
        print(f"Loading LoRA weights from {LORA_DIR}...")
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
        model = model.eval()
    else:
        print(f"WARNING: LoRA path {LORA_DIR} not found. Running with base model.")
        model = base_model.eval()
    
    # 3. Process Audio
    print(f"\nAnalyzing Audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Total Duration: {duration:.2f}s")
    
    all_cluster_tokens = []
    
    # 4. Sliding Window Inference
    print("\nRunning Director Inference...")
    for win_start in np.arange(0, duration, CHUNK_SEC):
        win_end = win_start + CHUNK_SEC
        if win_end > duration:
            if (duration - win_start) < (CHUNK_SEC * 0.5):
                break
            win_end = duration
            
        start_idx = int(win_start * sr)
        end_idx   = int(win_end * sr)
        y_chunk = y[start_idx:end_idx]
        
        # Qwen's feature extractor handles padding if necessary, but we can pad just in case
        actual_len = len(y_chunk) / sr
        
        # Director Prompt
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this {round(actual_len, 1)}s audio segment. "
            f"The difficulty is {difficulty}. "
            "Predict the ordered sequence of rhythmic pattern cluster tokens "
            "that best matches the audio's energy, density, and rhythm."
        )
        
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = processor(
            text=text,
            audio=[y_chunk],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        # Decode prediction
        input_length = inputs["input_ids"].shape[1]
        response = processor.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        tokens = extract_tokens_from_response(response)
        
        print(f"  [{win_start:04.1f}s - {win_end:04.1f}s]: {' '.join(tokens)}")
        all_cluster_tokens.extend(tokens)
        
    # 5. Actor Translation
    print("\nRunning Actor Translation to Physical Measures...")
    physical_measures = []
    for t in all_cluster_tokens:
        if t in cluster_dict and cluster_dict[t]:
            pattern_str = random.choice(cluster_dict[t])
            physical_measures.append(pattern_str)
        else:
            physical_measures.append("\n".join(["0000" for _ in range(192)]))
            
    # 6. SSC Formatting
    print("\nFormatting SSC File...")
    ssc_header = f"""#VERSION:0.83;
#TITLE:Generated Qwen Hierarchical Chart;
#SUBTITLE:;
#ARTIST:Qwen Director x Actor SubDecoder;
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
{",\n".join(physical_measures)}
;"""

    with open(out_ssc_path, "w", encoding="utf-8") as f:
        f.write(ssc_header + ssc_chart)
        
    print(f"\n==========================================")
    print(f"✅ Qwen Hierarchical Beatmap Generated!")
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
