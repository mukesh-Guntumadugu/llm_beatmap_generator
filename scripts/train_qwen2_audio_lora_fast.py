import os
import torch
import librosa
from datasets import load_dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List, Any

# CONFIGURATION
MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"  # Changed from HuggingFace to your local cluster path
DATASET_PATH = os.environ.get(
    "DATASET_OVERRIDE",
    "/data/mg546924/llm_beatmap_generator/sft_dataset/dataset.jsonl"
)
OUTPUT_DIR = "/data/mg546924/models/qwen2-audio-lora-onsets-fast"
BLOCK_SIZE = 512 # max text tokens

def main():
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, fix_mistral_regex=True)
    
    # 1. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}")
    full_dataset = load_dataset('json', data_files={'train': DATASET_PATH})['train']
    
    # Split 5% of data for validation evaluating (~50 songs out of 1000)
    dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train dataset size: {len(dataset['train'])} | Validation dataset size: {len(dataset['test'])}")
    
    # Save the list of validation songs so we know what they are
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "validation_songs.txt"), "w") as f:
        f.write("Songs used for validation (eval_loss):\n")
        f.write("-" * 40 + "\n")
        for item in dataset['test']:
            audio_url = item['messages'][0]['content'][0]['audio_url']
            song_name = audio_url.split("/")[-2] if "/" in audio_url else audio_url
            f.write(f"{song_name}\n")
    print(f"Saved list of validation songs to {os.path.join(OUTPUT_DIR, 'validation_songs.txt')}")

    # 2. Preprocess function
    def preprocess_function(examples):
        # We need to process each conversation
        input_ids = []
        attention_mask = []
        labels = []
        audio_features = []
        feature_attention_mask = []
        
        for messages in examples['messages']:
            # Load the audio for the user message
            user_msg = messages[0]
            audio_url = user_msg['content'][0]['audio_url'] # Assuming audio in first element
            
            # --- FIX MAC PATHS FOR CLUSTER ---
            # The dataset.jsonl was generated locally so it contains /Users/mukesh paths.
            # We dynamically replace it here so it works on the cluster.
            if "/Users/mukeshguntumadugu/" in audio_url:
                audio_url = audio_url.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")
            # ---------------------------------
            
            # Load audio using librosa
            y, sr = librosa.load(audio_url, sr=processor.feature_extractor.sampling_rate)
            
            # BYPASS QWEN REGEX BUG: The built-in apply_chat_template replaces the word "audio" with <|AUDIO|> 
            # causing crash if dataset text contains "audio_url" or "audio segment". We build the string manually:
            user_text = messages[0]['content'][1]['text']
            assistant_text = messages[1]['content'][0]['text']
            
            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
            )
            
            # Use processor to get inputs
            inputs = processor(
                text=text,
                audio=[y],
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt"
            )
            
            # Label masking (only predict assistant part for SFT)
            # A common simplified approach is to train on the whole sequence, but ideally we mask out the user part.
            # For simplicity, if we pass input_ids as labels, it trains on the whole sequence.
            label = inputs["input_ids"].clone()
            
            input_ids.append(inputs["input_ids"][0])
            attention_mask.append(inputs["attention_mask"][0])
            labels.append(label[0])
            
            # Audio features might need to be extracted from processor outputs
            if "audio_features" in inputs:
                 audio_features.append(inputs["audio_features"][0])
            elif "input_features" in inputs:
                 audio_features.append(inputs["input_features"][0])
                 
            if "feature_attention_mask" in inputs:
                 feature_attention_mask.append(inputs["feature_attention_mask"][0])

        # Return dict of lists
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
    # Map preprocess over dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1, # process 1 by 1 due to audio loading
        remove_columns=dataset["train"].column_names,
    )

    # Note: custom DataCollator is needed to pad the sequences properly in a batch
    # PyTorch's default collate won't work perfectly on list of tensors of varying sizes.
    @dataclass
    class MultimodalDataCollator:
        processor: Any
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Convert dataset lists back to PyTorch tensors
            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # -100 to ignore in loss
            
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

    # 3. Model Loading with QLoRA (4-bit)
    print("Loading Base Model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # 4. LoRA Adapter Config
    print("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target attention layers
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Keep small due to large model
        gradient_accumulation_steps=8,  # Simulate larger batch size
        per_device_eval_batch_size=1,   # Eval batch size 1
        optim="paged_adamw_32bit",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoints aligned with evaluation
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # Save loss to TensorBoard logs
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Use bfloat16 for stability
        max_grad_norm=0.3,
        num_train_epochs=5, # Force training for exactly 5 epochs for fast iteration
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # Disable wandb for now
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
    
    print("Saving final model adapter (Epoch 5)...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Done. Model adapter saved to {OUTPUT_DIR}")

    # 7. Extract Loss History and Save Chart/CSV
    print("Generating training vs validation loss chart and CSVs...")
    log_history = trainer.state.log_history

    train_logs = [log for log in log_history if "loss" in log and "epoch" in log]
    eval_logs = [log for log in log_history if "eval_loss" in log and "epoch" in log]

    import csv
    if train_logs:
        with open(os.path.join(OUTPUT_DIR, "training_loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss", "learning_rate"])
            for log in train_logs:
                writer.writerow([round(log.get("epoch", 0), 2), log.get("step"), log.get("loss"), log.get("learning_rate")])
                
    if eval_logs:
        with open(os.path.join(OUTPUT_DIR, "validation_loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "eval_loss"])
            for log in eval_logs:
                writer.writerow([round(log.get("epoch", 0), 2), log.get("step"), log.get("eval_loss")])

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        if train_logs:
            epochs = [log.get("epoch") for log in train_logs]
            losses = [log.get("loss") for log in train_logs]
            plt.plot(epochs, losses, label="Training Loss", color="blue", alpha=0.6)
            
        if eval_logs:
            e_epochs = [log.get("epoch") for log in eval_logs]
            e_losses = [log.get("eval_loss") for log in eval_logs]
            plt.plot(e_epochs, e_losses, label="Validation Loss", color="red", marker="o", linewidth=2)
        
        plt.title("Training vs Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, "loss_chart.png")
        plt.savefig(chart_path)
        print(f"Saved loss chart to {chart_path}")
    except ImportError:
        print("matplotlib not found in environment, skipping loss chart PNG generation.")
        
    print(f"Done. Saved CSVs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
