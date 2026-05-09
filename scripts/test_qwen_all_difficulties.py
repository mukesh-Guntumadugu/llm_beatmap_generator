#!/usr/bin/env python3
"""
test_qwen_all_difficulties.py
==============================
End-to-end test of the Qwen2-Audio Director → Actor pipeline.

Given an audio file + BPM, generates a complete .ssc beatmap with
ALL 5 StepMania difficulty levels in one file:
  Beginner | Easy | Medium | Hard | Challenge

Usage (on HPC):
  python scripts/test_qwen_all_difficulties.py \
      --audio /path/to/song.wav \
      --bpm 130.0 \
      --out output/test_song.ssc
"""

import os
import sys
import json
import re
import glob
import random
import argparse

import torch
import librosa
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# ── Paths ──
DICT_PATH  = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns.json"
BASE_MODEL = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR   = "/data/mg546924/models/qwen2-audio-hierarchical-director"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"

CHUNK_SEC = 5.0

# StepMania difficulty levels with their meter (note density rating 1-10)
DIFFICULTIES = [
    ("Beginner",  1),
    ("Easy",      3),
    ("Medium",    5),
    ("Hard",      7),
    ("Challenge", 9),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_actor_dictionary(dict_path):
    print(f"Loading Actor Sub-Decoder dictionary from {dict_path}...")
    with open(dict_path, "r") as f:
        cluster_dict = json.load(f)
    print(f"  Loaded {len(cluster_dict)} physical cluster mappings.")
    return cluster_dict


def extract_cluster_tokens(text):
    """Pull <|cluster_N|> tokens from raw model output."""
    return re.findall(r"<\|cluster_\d+\|>", text)

def align_tokens_to_measures(tokens, target_len):
    """Stretches or compresses the sequence of tokens to perfectly match target_len."""
    if not tokens:
        return ["<|cluster_0|>"] * target_len
    aligned = []
    for i in range(target_len):
        idx = int(i * len(tokens) / target_len)
        aligned.append(tokens[idx])
    return aligned


def tokens_to_measures(tokens, cluster_dict, difficulty_name):
    """
    Actor: translate cluster tokens → physical .ssc measure strings.
    Harder difficulties get denser patterns from the cluster dict.
    """
    difficulty_bias = {
        "Beginner":  0,
        "Easy":      1,
        "Medium":    2,
        "Hard":      3,
        "Challenge": 4,
    }
    bias = difficulty_bias.get(difficulty_name, 2)

    measures = []
    for t in tokens:
        if t in cluster_dict and cluster_dict[t]:
            patterns = cluster_dict[t]
            def density(p):
                return sum(1 for row in p.split("\n") if any(c != "0" for c in row))
            sorted_patterns = sorted(patterns, key=density)
            idx = min(bias, len(sorted_patterns) - 1)
            measures.append(sorted_patterns[idx])
        else:
            measures.append("\n".join(["0000"] * 16))
    return measures


def format_ssc_chart(difficulty_name, meter, measures_str):
    return f"""
//---------------dance-single - {difficulty_name}----------------
#NOTEDATA:;
#CHARTNAME:{difficulty_name};
#STEPSTYPE:dance-single;
#DESCRIPTION:{difficulty_name};
#CHARTSTYLE:;
#DIFFICULTY:{difficulty_name};
#METER:{meter};
#RADARVALUES:0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000;
#CREDIT:Qwen2-Audio-Director;
#NOTES:
{measures_str}
;"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True,             help="Path to input audio file (.wav/.mp3/.ogg)")
    parser.add_argument("--bpm",    type=float, default=130.0, help="Song BPM")
    parser.add_argument("--out",    default="output/test_qwen_beatmap.ssc", help="Output .ssc path")
    parser.add_argument("--offset", type=float, default=0.0,   help="Chart offset in seconds")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # ── Load Actor Dictionary ──
    cluster_dict = load_actor_dictionary(DICT_PATH)

    # ── Load Qwen Model (load once, reuse for all difficulties) ──
    print("Loading Qwen Processor...")
    try:
        processor = AutoProcessor.from_pretrained(LORA_DIR, trust_remote_code=True)
    except Exception:
        print("Fallback to base processor...")
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        with open(TOKENS_TXT, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    print("Loading Base Model in 4-bit quantization...")
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
    base_model.resize_token_embeddings(len(processor.tokenizer))

    print(f"Loading LoRA weights from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR).eval()
    print("✅ Qwen Director ready!")

    # ── Load + Chunk Audio ──
    print(f"\nLoading audio: {args.audio}")
    sr_target = processor.feature_extractor.sampling_rate
    y_full, _ = librosa.load(args.audio, sr=sr_target)
    duration   = librosa.get_duration(y=y_full, sr=sr_target)
    print(f"Duration: {duration:.2f}s  |  BPM: {args.bpm}")

    # Pre-slice audio into chunks (shared across all difficulty runs)
    audio_chunks = []
    for win_start in np.arange(0, duration, CHUNK_SEC):
        win_end = win_start + CHUNK_SEC
        if win_end > duration:
            if (duration - win_start) < CHUNK_SEC * 0.5:
                break
            win_end = duration
        s = int(win_start * sr_target)
        e = int(win_end   * sr_target)
        audio_chunks.append((win_start, win_end, y_full[s:e]))
    print(f"Audio sliced into {len(audio_chunks)} x 5s chunks.\n")

    # ── SSC Header ──
    song_title = os.path.splitext(os.path.basename(args.audio))[0]
    ssc_header = f"""#VERSION:0.83;
#TITLE:{song_title};
#SUBTITLE:;
#ARTIST:AI Generated;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#ORIGIN:;
#CREDIT:Qwen2-Audio Director;
#BANNER:;
#BACKGROUND:;
#PREVIEWVID:;
#JACKET:;
#CDIMAGE:;
#DISCIMAGE:;
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:{os.path.basename(args.audio)};
#OFFSET:{args.offset:.6f};
#SAMPLESTART:0.000000;
#SAMPLELENGTH:10.000000;
#SELECTABLE:YES;
#BPMS:0.000000={args.bpm:.6f};
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

    all_charts = ""

    # ── Run Director ONCE (shared across all difficulties) ──
    print("=" * 55)
    print("Running Qwen Director Inference (once for all difficulties)")
    print("=" * 55)

    all_tokens = []
    for (win_start, win_end, y_chunk) in audio_chunks:
        prompt = (
            f"Listen to this {round(win_end - win_start, 1)}s audio segment. "
            f"Song BPM: {args.bpm:.1f}. "
            "Predict the ordered sequence of rhythmic pattern cluster tokens."
        )
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = processor(
            text=text,
            audio=[y_chunk],
            sampling_rate=sr_target,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        input_length = inputs["input_ids"].shape[1]
        response = processor.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=False)
        tokens   = extract_cluster_tokens(response)
        print(f"  [{win_start:05.1f}s-{win_end:05.1f}s] raw='{response[:100].strip()}' → {tokens if tokens else '(none)'}")
        all_tokens.extend(tokens)

    print(f"\n  Total cluster tokens predicted: {len(all_tokens)}\n")

    # ── Calculate Mathematical Target Measures ──
    total_beats = duration * (args.bpm / 60.0)
    target_measures = int(np.round(total_beats / 4.0)) # 4/4 time
    print(f"Math: {duration:.2f}s at {args.bpm} BPM = {total_beats:.2f} beats = {target_measures} physical measures")
    
    print(f"Aligning {len(all_tokens)} generated tokens to {target_measures} measures...")
    aligned_tokens = align_tokens_to_measures(all_tokens, target_measures)

    # ── Actor: Build Each Difficulty from the SAME token sequence ──
    for diff_name, meter in DIFFICULTIES:
        print(f"{'='*55}")
        print(f"Building difficulty: {diff_name.upper()} (meter {meter})")

        measures = tokens_to_measures(aligned_tokens, cluster_dict, diff_name)
        measures_str = ",\n".join(measures)
        all_charts += format_ssc_chart(diff_name, meter, measures_str)
        print(f"  ✅ {diff_name}: {len(measures)} measures generated")

    # ── Write Final .ssc ──
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(ssc_header + all_charts)

    print(f"\n{'='*55}")
    print(f"✅ Full Qwen beatmap written to: {args.out}")
    print(f"   Difficulties: {', '.join(d[0] for d in DIFFICULTIES)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
