#!/usr/bin/env python3
"""
train_flamingo_5s_lora.py
=========================
Fine-tunes Music-Flamingo on 5-second onset detection chunks.
Uses huggingface_hub to download weights locally first, then loads
via AutoModelForCausalLM (compatible with Python 3.9 / transformers 4.x).
"""
import os
import json
import torch
import librosa
import csv as csv_mod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download

# ── Paths ──
HF_MODEL_ID = "nvidia/music-flamingo-hf"
LOCAL_MODEL_PATH = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints/model_weights"

# Point HF cache to shared /data path so compute nodes can find cached modules
os.environ['HF_HOME'] = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints"
DATASET_JSONL = "/data/mg546924/llm_beatmap_generator/sft_dataset_5s_chunks/dataset.jsonl"
OUTPUT_DIR = "/data/mg546924/models/music-flamingo-lora-onsets"

NUM_EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LENGTH = 512

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    # ── Step 1: Download weights if not already cached ──
    config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"Downloading Music-Flamingo weights from {HF_MODEL_ID}...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=LOCAL_MODEL_PATH,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print("✅ Download complete!")
    else:
        print(f"✅ Weights already at {LOCAL_MODEL_PATH}, skipping download.")

    # ── DEBUGGING: Print directory contents and config.json ──
    print(f"\n--- DEBUG: Contents of {LOCAL_MODEL_PATH} ---")
    if os.path.exists(LOCAL_MODEL_PATH):
        for f in os.listdir(LOCAL_MODEL_PATH):
            print(f)
        
        config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            print("\n--- DEBUG: config.json ---")
            with open(config_path, "r") as f:
                print(f.read())
        else:
            print("config.json not found!")
    else:
        print("Directory not found!")
        
    print("\n--- DEBUG: Checking for audio_flamingo modules ---")
    try:
        from audio_flamingo.models.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration
        print("Successfully imported from audio_flamingo!")
    except Exception as e:
        print(f"Failed to import from audio_flamingo: {e}")
        
    sys.exit(0)
    # ---------------------------------------------------------

    model.gradient_checkpointing_enable()

    print("Injecting LoRA adapters (r=16)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ──
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
            prompt = item["text_prompt"]
            response = str(item["text_response"])
            text = f"User: {prompt}\nAssistant: {response}"

            tok = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)

            # Mask prompt from labels
            prompt_text = f"User: {prompt}\nAssistant: "
            prompt_len = len(self.tokenizer(prompt_text).input_ids)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    print("Loading onset dataset...")
    full_dataset = OnsetDataset(DATASET_JSONL, tokenizer, MAX_LENGTH)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.001,
    )

    # ── Training Loop ──
    loss_log = []
    print(f"\n{'='*60}\nStarting Music-Flamingo Training: {NUM_EPOCHS} Epochs\n{'='*60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()
                if (batch_idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += outputs.loss.item()
                num_batches += 1
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Step {batch_idx+1}: loss={epoch_loss/num_batches:.4f}")
            except Exception as e:
                print(f"  Skipping batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue

        avg_train = epoch_loss / max(num_batches, 1)

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    out = model(
                        input_ids=batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                        labels=batch["labels"].to(model.device),
                    )
                    val_loss += out.loss.item()
                    val_batches += 1
                except:
                    continue

        avg_val = val_loss / max(val_batches, 1)
        print(f"\n Epoch {epoch+1}/{NUM_EPOCHS} — Train: {avg_train:.4f} | Val: {avg_val:.4f}\n")
        loss_log.append({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})

        if epoch + 1 == NUM_EPOCHS:
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
            model.save_pretrained(save_path)
            print(f"   Saved: {save_path}")

    pass

    print(f"\n Music-Flamingo Training Complete! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
