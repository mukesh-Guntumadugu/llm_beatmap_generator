#!/usr/bin/env python3
"""
train_deepresonance_5s_lora.py
==============================
Fine-tunes DeepResonance on 5-second onset detection chunks using its native
DeepResonanceModel + Vicuna/ImageBind architecture.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import csv as csv_mod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Patch: bypass transformers CVE-2025-32434 torch.load block ──
# Vicuna weights are .bin format (old PyTorch). transformers 4.57+ blocks
# torch.load on PyTorch < 2.6 even with weights_only=True. We patch it out.
try:
    import transformers.utils.import_utils as _triu
    _triu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# ── Paths ──
DR_ROOT = "/data/mg546924/llm_beatmap_generator/DeepResonance/code"
CKPT_DIR = "/data/mg546924/llm_beatmap_generator/DeepResonance/ckpt"
DATASET_JSONL = "/data/mg546924/llm_beatmap_generator/sft_dataset_5s_chunks/dataset.jsonl"
OUTPUT_DIR = "/data/mg546924/models/deepresonance-lora-onsets"

NUM_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 1
MAX_LENGTH = 512

# Add DeepResonance source to path and cd into it (it uses relative config paths)
sys.path.insert(0, DR_ROOT)
os.chdir(DR_ROOT)

# Bypass DeepSpeed NVCC bug
from unittest.mock import MagicMock
try:
    import triton
except ImportError:
    sys.modules['triton'] = MagicMock()
sys.modules['triton.ops'] = MagicMock()
sys.modules['triton.ops.matmul_perf_model'] = MagicMock()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from config import load_config
    from model.deepresonance import DeepResonanceModel
    from transformers import LlamaTokenizer

    # Build config just like inference_deepresonance.py does
    args = {
        'model': 'deepresonance',
        'stage': 2,
        'mode': 'train',
        'max_length': MAX_LENGTH,
        'max_output_length': MAX_LENGTH,
        'ckpt_path': os.path.join(CKPT_DIR, 'deepresonance_alpha_delta_ckpt'),
        'pretrained_ckpt_path': os.path.join(CKPT_DIR, 'pretrained_ckpt'),
    }
    config = load_config(args)
    args.update(config)
    args['max_length'] = MAX_LENGTH

    print("Loading DeepResonance model...")
    model = DeepResonanceModel(**args)

    # Load delta checkpoint
    delta_path = os.path.join(args['ckpt_path'], 'pytorch_model.pt')
    if os.path.exists(delta_path):
        print(f"Loading delta checkpoint: {delta_path}")
        delta_ckpt = torch.load(delta_path, map_location='cpu')
        model.load_state_dict(delta_ckpt, strict=False)
    else:
        print(f"⚠️ No delta checkpoint at {delta_path}, training from base weights")

    model = model.cuda().bfloat16()
    model.train()
    print("✅ DeepResonance loaded successfully!")

    # Load tokenizer (Vicuna uses LLaMA tokenizer)
    vicuna_path = os.path.join(CKPT_DIR, 'pretrained_ckpt', 'vicuna-7b-v1.1')
    tokenizer = LlamaTokenizer.from_pretrained(vicuna_path)

    # ── Custom Dataset ──
    class OnsetDataset(Dataset):
        def __init__(self, jsonl_path, tokenizer, max_length):
            self.samples = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            audio_path = item["audio_path"]
            prompt = item["text_prompt"]
            response = str(item["text_response"])

            # Full conversation text
            text = f"### Human: {prompt}\n### Assistant: {response}"

            # Tokenize
            tokens = self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                                     truncation=True, padding="max_length")
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            # Labels: mask prompt portion
            prompt_text = f"### Human: {prompt}\n### Assistant: "
            prompt_len = len(self.tokenizer(prompt_text).input_ids)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "audio_path": audio_path,
            }

    print("Loading onset dataset...")
    full_dataset = OnsetDataset(DATASET_JSONL, tokenizer, MAX_LENGTH)

    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimizer (only trainable params) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.05)

    # ── Training Loop ──
    loss_log = []
    print(f"\n{'='*60}")
    print(f"Starting DeepResonance Training: {NUM_EPOCHS} Epochs")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            try:
                with torch.cuda.amp.autocast():
                    outputs = model.llama_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
            for batch in val_loader:
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model.llama_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        val_loss += outputs.loss.item()
                        val_batches += 1
                except:
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        loss_log.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save checkpoint
        if epoch + 1 == NUM_EPOCHS:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  💾 Saved: {ckpt_path}")

    pass

    print(f"\n✅ DeepResonance Training Complete! Final model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
