#!/usr/bin/env python3
"""
test_mumu_all_difficulties.py
==============================
End-to-end test of the MuMu Director → Actor pipeline.

Given an audio file + BPM, generates a complete .ssc beatmap with
ALL 5 StepMania difficulty levels in one file:
  Beginner | Easy | Medium | Hard | Challenge

Usage (on HPC):
  python scripts/test_mumu_all_difficulties.py \
      --audio /path/to/song.wav \
      --bpm 130.0 \
      --out output/test_song.ssc

The output .ssc can be dropped directly into StepMania.
"""

import os
import sys
import json
import re
import glob
import random
import argparse

import torch
import torchaudio
import numpy as np

# ── Paths ──
MUMU_ROOT  = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/MuMu-LLaMA"
DICT_PATH  = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns.json"
MODELS_DIR = "/data/mg546924/models/mumu-hierarchical-director"

sys.path.insert(0, MUMU_ROOT)

SAMPLE_RATE = 24000
CHUNK_SEC   = 5.0
MAX_WORDS   = 256

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
        "Beginner":  0,   # pick sparse patterns
        "Easy":      1,
        "Medium":    2,
        "Hard":      3,
        "Challenge": 4,   # pick dense patterns
    }
    bias = difficulty_bias.get(difficulty_name, 2)

    measures = []
    for t in tokens:
        if t in cluster_dict and cluster_dict[t]:
            patterns = cluster_dict[t]
            # Sort by note density (count non-zero rows) and bias toward
            # denser or sparser patterns based on difficulty
            def density(p):
                return sum(1 for row in p.split("\n") if any(c != "0" for c in row))
            sorted_patterns = sorted(patterns, key=density)
            # Clamp index to valid range
            idx = min(bias, len(sorted_patterns) - 1)
            measures.append(sorted_patterns[idx])
        else:
            # Fallback: empty measure
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
#CREDIT:MuMu-Director;
#NOTES:
{measures_str}
;"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  required=True,          help="Path to input audio file (.wav/.mp3/.ogg)")
    parser.add_argument("--bpm",    type=float, default=130.0, help="Song BPM")
    parser.add_argument("--out",    default="output/test_beatmap.ssc", help="Output .ssc path")
    parser.add_argument("--offset", type=float, default=0.0,  help="Chart offset in seconds")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # ── Load Actor Dictionary ──
    cluster_dict = load_actor_dictionary(DICT_PATH)

    # ── Load MuMu Model (load once, reuse for all difficulties) ──
    from llama.mumu_llama import MuMu_LLaMA
    from transformers import LlamaTokenizer
    import llama

    print("Loading MuMu Director tokenizer...")
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    # Inject the 2189 cluster tokens — same as done during training
    TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
    if os.path.exists(TOKENS_TXT):
        with open(TOKENS_TXT) as f:
            cluster_tokens = [t.strip() for t in f if t.strip()]
        added = tokenizer.add_special_tokens({'additional_special_tokens': cluster_tokens})
        print(f"  Added {added} cluster tokens to tokenizer (total vocab: {len(tokenizer)})")
    else:
        print(f"  WARNING: {TOKENS_TXT} not found — cluster tokens not injected!")

    # ── Monkey-Patch Tokenizer Decode to Prevent SentencePiece Crashes ──
    original_decode = tokenizer.decode
    def safe_decode(token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
            
        res = []
        for tid in token_ids:
            if tid in tokenizer.added_tokens_decoder:
                res.append(str(tokenizer.added_tokens_decoder[tid]))
            else:
                try:
                    if tid < tokenizer.sp_model.get_piece_size():
                        res.append(original_decode([tid], **kwargs))
                except Exception:
                    pass
        return "".join(res)
    
    tokenizer.decode = safe_decode

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
    # MuMu uses a custom Transformer — resize embeddings manually
    import torch.nn as nn
    new_vocab = len(tokenizer)
    old_emb = model.llama.tok_embeddings
    old_size, dim = old_emb.weight.shape
    if new_vocab != old_size:
        new_emb = nn.Embedding(new_vocab, dim)
        new_emb.weight.data[:old_size] = old_emb.weight.data
        model.llama.tok_embeddings = new_emb
        if hasattr(model.llama, 'output') and hasattr(model.llama.output, 'weight'):
            new_out = nn.Linear(dim, new_vocab, bias=False)
            new_out.weight.data[:old_size] = model.llama.output.weight.data
            model.llama.output = new_out

    # Load best checkpoint — prefer checkpoint_final.pth, fall back to latest epoch
    final_ckpt = os.path.join(MODELS_DIR, "checkpoint_final.pth")
    epoch_ckpts = sorted(glob.glob(os.path.join(MODELS_DIR, "checkpoint_epoch*.pth")))
    if os.path.exists(final_ckpt):
        latest_ckpt = final_ckpt
    elif epoch_ckpts:
        latest_ckpt = epoch_ckpts[-1]
    else:
        print("ERROR: No MuMu checkpoints found in", MODELS_DIR)
        sys.exit(1)
    print(f"Loading weights: {latest_ckpt}")
    state_dict = torch.load(latest_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().bfloat16().eval()
    print("✅ MuMu Director ready!")

    # ── Load + Chunk Audio ──
    print(f"\nLoading audio: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    audio_mono = torch.mean(waveform, 0)
    duration   = audio_mono.shape[0] / SAMPLE_RATE
    print(f"Duration: {duration:.2f}s  |  BPM: {args.bpm}")

    # Pre-slice audio into chunks (shared across all difficulty runs)
    audio_chunks = []
    for win_start in np.arange(0, duration, CHUNK_SEC):
        win_end = win_start + CHUNK_SEC
        if win_end > duration:
            if (duration - win_start) < CHUNK_SEC * 0.5:
                break
            win_end = duration
        s = int(win_start * SAMPLE_RATE)
        e = int(win_end   * SAMPLE_RATE)
        chunk = audio_mono[s:e]
        if chunk.shape[0] < int(CHUNK_SEC * SAMPLE_RATE):
            pad = int(CHUNK_SEC * SAMPLE_RATE) - chunk.shape[0]
            chunk = torch.cat([chunk, torch.zeros(pad)])
        audio_chunks.append((win_start, win_end, chunk))
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
#CREDIT:MuMu-LLaMA Director;
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
    difficulty_clusters_export = {}

    # ── Run Director for Each Difficulty ──
    for diff_name, meter in DIFFICULTIES:
        print(f"{'='*55}")
        print(f"Generating difficulty: {diff_name.upper()} (meter {meter})")
        print(f"{'='*55}")

        all_tokens = []
        for (win_start, win_end, chunk) in audio_chunks:
            prompt = (
                f"You are a rhythm game beatmap generator. "
                f"Listen to this {CHUNK_SEC:.0f}-second audio segment. "
                f"The song BPM is {args.bpm:.1f}. "
                f"Difficulty: {diff_name}. "
                f"Predict the ordered sequence of rhythmic pattern cluster tokens."
            )
            input_text = llama.utils.format_prompt(prompt)
            input_ids  = tokenizer(input_text).input_ids
            input_ids  = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).cuda()
            audio_t    = chunk.cuda().float()  # keep 1D [T]; MuMu's forward_audio adds batch dim internally

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output_ids = model.generate(
                        input_ids,
                        audios=audio_t,
                        max_gen_len=20,
                        temperature=0.7,
                        top_p=0.9,
                    )
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            tokens   = extract_cluster_tokens(response)
            print(f"  [{win_start:05.1f}s-{win_end:05.1f}s] raw='{response[:120]}' → {tokens if tokens else '(none)'}")
            # Fallback: if model produces no cluster tokens, sample random ones from dictionary
            if not tokens:
                available = list(cluster_dict.keys())
                tokens = [random.choice(available) for _ in range(4)]  # 4 clusters per 5s chunk
                print(f"    ↳ fallback random clusters: {tokens}")
            all_tokens.extend(tokens)

        print(f"  Total cluster tokens: {len(all_tokens)}")

        # ── Calculate Mathematical Target Measures ──
        total_beats = duration * (args.bpm / 60.0)
        target_measures = int(np.round(total_beats / 4.0)) # 4/4 time
        aligned_tokens = align_tokens_to_measures(all_tokens, target_measures)

        # Actor: tokens → physical measures
        measures = tokens_to_measures(aligned_tokens, cluster_dict, diff_name)
        measures_str = ",\n".join(measures)

        # Build this chart block
        all_charts += format_ssc_chart(diff_name, meter, measures_str)
        print(f"  ✅ {diff_name}: {len(measures)} measures generated\n")
        
        difficulty_clusters_export[diff_name] = aligned_tokens

    # ── Write Final .ssc ──
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(ssc_header + all_charts)

    # ── Export Cluster Choices to JSON ──
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(args.out)[0]
    cluster_export_path = f"{base_name}_cluster_picked_{timestamp}.json"
    with open(cluster_export_path, "w", encoding="utf-8") as f:
        json.dump(difficulty_clusters_export, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Full beatmap written to: {args.out}")
    print(f"   Difficulties: {', '.join(d[0] for d in DIFFICULTIES)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
