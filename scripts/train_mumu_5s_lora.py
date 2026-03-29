#!/usr/bin/env python3
"""
train_mumu_5s_lora.py
=====================
Fine-tunes MuMu-LLaMA on 5-second onset detection chunks using its native
Accelerate-based training loop (from the MuMu-LLaMA repo).
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import csv as csv_mod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Paths ──
MUMU_ROOT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/MuMu-LLaMA"
LLAMA_DIR = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA"
MUMU_CKPT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/MuMu-LLaMA-MusicGen/checkpoint.pth"
DATASET_JSONL = "/data/mg546924/llm_beatmap_generator/sft_dataset_5s_chunks/dataset.jsonl"
OUTPUT_DIR = "/data/mg546924/models/mumu-llama-lora-onsets"

NUM_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 1
MAX_WORDS = 512
SAMPLE_RATE = 24000

# Add MuMu source to path
sys.path.insert(0, MUMU_ROOT)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import MuMu's native model and tokenizer utils
    from llama.mumu_llama import MuMu_LLaMA
    import llama
    from transformers import LlamaTokenizer

    print("Loading LLaMA tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)

    import argparse
    model_args = argparse.Namespace(
        mert_path="m-a-p/MERT-v1-330M",
        vit_path="google/vit-base-patch16-224",
        vivit_path="google/vivit-b-16x2-kinetics400",
        music_decoder="musicgen",
        music_decoder_path="facebook/musicgen-small",
        max_words=MAX_WORDS,
    )

    print("Loading MuMu-LLaMA model...")
    model = MuMu_LLaMA(
        llama_ckpt_dir=os.path.join(LLAMA_DIR, "7B"),
        llama_tokenizer=LLAMA_DIR,
        model_args=model_args,
        knn_dir="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts",
        stage=3,  # Stage 3 = QA/instruction following mode
    )

    # Load the pretrained MuMu checkpoint
    print(f"Loading pretrained checkpoint: {MUMU_CKPT}")
    ckpt = torch.load(MUMU_CKPT, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().float()
    print("✅ MuMu-LLaMA loaded successfully!")

    # ── Custom Dataset for 5s onset chunks ──
    class OnsetDataset(Dataset):
        def __init__(self, jsonl_path, tokenizer, max_words):
            self.samples = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
            self.tokenizer = tokenizer
            self.max_words = max_words

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            audio_path = item["audio_path"]
            prompt = item["text_prompt"]
            response = str(item["text_response"])

            # Load audio at MuMu's expected 24kHz
            waveform, sr = torchaudio.load(audio_path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = torch.mean(waveform, 0)  # mono

            # Tokenize using MuMu's format
            format_instruction = prompt
            input1 = llama.utils.format_prompt(format_instruction)
            input2 = input1 + response

            input1_ids = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
            input2_ids = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)

            # Pad or truncate
            padding = self.max_words - input2_ids.shape[0]
            if padding < 0:
                input2_ids = input2_ids[:self.max_words]
            elif padding > 0:
                input2_ids = torch.cat([input2_ids, torch.zeros(padding, dtype=torch.int64)])

            # Labels: mask the prompt portion
            labels = input2_ids.clone()
            labels[:len(input1_ids)] = 0

            # Attention mask
            mask = input2_ids.ne(0)

            return input2_ids, labels, mask, audio, "Audio", ""

    print("Loading onset dataset...")
    full_dataset = OnsetDataset(DATASET_JSONL, tokenizer, MAX_WORDS)

    # 95/5 train/val split
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.05
    )

    # ── Training Loop ──
    loss_log = []
    print(f"\n{'='*60}")
    print(f"Starting MuMu-LLaMA Training: {NUM_EPOCHS} Epochs")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (examples, labels, mask, audio, modality, caption) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            examples = examples.cuda()
            labels = labels.cuda()
            audio = audio.cuda()

            try:
                with torch.cuda.amp.autocast():
                    c_loss, m_loss = model(examples, labels, audios=audio, music_caption=None)
                    loss = c_loss + m_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    avg = epoch_loss / num_batches
                    print(f"  Step {batch_idx+1}: loss={avg:.4f}")

            except Exception as e:
                print(f"  Skipping batch {batch_idx}: {e}")
                continue

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for examples, labels, mask, audio, modality, caption in val_loader:
                examples = examples.cuda()
                labels = labels.cuda()
                audio = audio.cuda()
                try:
                    with torch.cuda.amp.autocast():
                        c_loss, m_loss = model(examples, labels, audios=audio, music_caption=None)
                        val_loss += (c_loss + m_loss).item()
                        val_batches += 1
                except:
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        loss_log.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save checkpoint each epoch
        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  💾 Saved: {ckpt_path}")

    # ── Save loss CSV ──
    with open(os.path.join(OUTPUT_DIR, "mumu_training_log.csv"), "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for entry in loss_log:
            writer.writerow([entry["epoch"], f"{entry['train_loss']:.4f}", f"{entry['val_loss']:.4f}"])

    print(f"\n✅ MuMu-LLaMA Training Complete! Final model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
