#!/usr/bin/env python3
import os
import sys

# Prevent accelerate from triggering DeepSpeed's buggy nvcc compiler check
sys.modules['deepspeed'] = None

import torch
import librosa
from datasets import load_dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List, Any

# CONFIGURATION
MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct" 
DATASET_PATH = "/data/mg546924/llm_beatmap_generator/hierarchical_sft_dataset/hierarchical_train.jsonl"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
OUTPUT_DIR = "/data/mg546924/models/qwen2-audio-hierarchical-director"
BLOCK_SIZE = 512
MAX_AUDIO_DURATION_SEC = 10  # Hard cap: prevents giant mel spectrograms from hanging forward pass
MAX_SEQ_LENGTH = 512         # Hard cap: prevents OOM from very long tokenized sequences

def load_cluster_tokens(tokens_txt_path):
    if not os.path.exists(tokens_txt_path):
        raise FileNotFoundError(f"Token list not found at {tokens_txt_path}.")
    with open(tokens_txt_path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens

def main():
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, fix_mistral_regex=True)
    
    # 1. Extend Tokenizer
    print("Extending tokenizer with cluster tokens...")
    cluster_tokens = load_cluster_tokens(TOKENS_TXT)
    before_len = len(processor.tokenizer)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
    after_len = len(processor.tokenizer)
    print(f"Tokenizer vocab size: {before_len} -> {after_len} (+{after_len - before_len})")

    # 2. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}")
    full_dataset = load_dataset('json', data_files={'train': DATASET_PATH})['train']
    
    # Split 5% of data for validation
    dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train dataset size: {len(dataset['train'])} | Validation dataset size: {len(dataset['test'])}")
    
    # 3. Preprocess function
    def preprocess_function(examples):
        input_ids = []
        attention_mask = []
        labels = []
        audio_features = []
        feature_attention_mask = []
        
        for audio_url, text_prompt, text_response in zip(examples['audio_path'], examples['text_prompt'], examples['text_response']):
            
            # HPC path fix
            if "/Users/mukeshguntumadugu/" in audio_url:
                audio_url = audio_url.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")
            
            # We skip chunks if audio doesn't exist
            if not os.path.exists(audio_url):
                # Fallback to local audio for local testing
                if "local_audio_path" in examples and os.path.exists(examples["local_audio_path"]):
                    audio_url = examples["local_audio_path"]
                else:
                    continue # Skip
                    
            try:
                sr_target = processor.feature_extractor.sampling_rate
                # KEY FIX: cap audio duration to prevent enormous mel spectrograms
                # that cause the first forward pass to hang for hours
                y, sr = librosa.load(
                    audio_url,
                    sr=sr_target,
                    duration=MAX_AUDIO_DURATION_SEC
                )
                if len(y) == 0:
                    continue  # Skip silent/empty files
            except Exception:
                continue # Skip on load error
            
            # Format conversational template manually to avoid regex bug
            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{text_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{text_response}<|im_end|>\n"
            )
            
            inputs = processor(
                text=text,
                audio=[y],
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=False,
            )
            
            label = inputs["input_ids"].clone()
            
            # Mask out the user prompt so we only train on the cluster tokens
            # Find the start of the assistant response
            try:
                assistant_str = "<|im_start|>assistant\n"
                assistant_ids = processor.tokenizer.encode(assistant_str, add_special_tokens=False)
                # Simple approximation: mask the first N tokens before the actual response
                # But to be safe, we can just let it train on the whole sequence or mask correctly.
                # A common hack: just set user prompt labels to -100
                input_id_list = inputs["input_ids"][0].tolist()
                response_ids = processor.tokenizer.encode(text_response + "<|im_end|>\n", add_special_tokens=False)
                resp_len = len(response_ids)
                
                label[0, :-resp_len] = -100 # Mask everything before the response
            except Exception:
                pass # If it fails, train on full sequence (still works, just less ideal)
            
            input_ids.append(inputs["input_ids"][0])
            attention_mask.append(inputs["attention_mask"][0])
            labels.append(label[0])
            
            if "audio_features" in inputs:
                 audio_features.append(inputs["audio_features"][0])
            elif "input_features" in inputs:
                 audio_features.append(inputs["input_features"][0])
                 
            if "feature_attention_mask" in inputs:
                 feature_attention_mask.append(inputs["feature_attention_mask"][0])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_features" if len(audio_features) > 0 and "audio_features" in inputs else "input_features": audio_features
        }
        if len(feature_attention_mask) > 0:
            result["feature_attention_mask"] = feature_attention_mask
            
        return result

    print("Pre-processing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1, 
        remove_columns=dataset["train"].column_names,
    )

    @dataclass
    class MultimodalDataCollator:
        processor: Any
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            if len(features) == 0:
                return {}
            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            if "audio_features" in features[0]:
                batch["audio_features"] = torch.stack([torch.tensor(f["audio_features"], dtype=torch.float16) for f in features])
            elif "input_features" in features[0]:
                batch["input_features"] = torch.stack([torch.tensor(f["input_features"], dtype=torch.float16) for f in features])
                
            if "feature_attention_mask" in features[0]:
                batch["feature_attention_mask"] = torch.stack([torch.tensor(f["feature_attention_mask"], dtype=torch.long) for f in features])
                
            return batch

    # 3. Model Loading (Pure bfloat16, load on CPU first to safely resize embeddings)
    print("Loading Base Model in bfloat16 on CPU...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # RESIZE EMBEDDINGS BEFORE PEFT AND BEFORE MOVING TO GPU
    print("Resizing token embeddings on CPU...")
    model.resize_token_embeddings(after_len, mean_resizing=False)
    
    print("Moving model to GPU...")
    model = model.to("cuda")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # 4. LoRA Adapter Config
    print("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        # IMPORTANT: embed_tokens and lm_head must be trained since we added new cluster tokens!
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        logging_steps=1,           # Log every step so we can see per-step timing
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=50,             # Save checkpoint every 50 steps
        save_total_limit=2,        # Keep only last 2 checkpoints
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        num_train_epochs=5,
        warmup_ratio=0.03,
        group_by_length=False,     # Disable: was hiding bad samples by grouping by length
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
        data_collator=MultimodalDataCollator(processor),
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving final model adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
