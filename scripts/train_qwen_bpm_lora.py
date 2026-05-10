import os
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
DATASET_PATH = "/data/mg546924/llm_beatmap_generator/bpm_sft_dataset/bpm_dataset.jsonl"
OUTPUT_DIR = "/data/mg546924/models/qwen-bpm-lora"

def main():
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, fix_mistral_regex=True)
    
    print(f"Loading dataset from {DATASET_PATH}")
    full_dataset = load_dataset('json', data_files={'train': DATASET_PATH})['train']
    dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train dataset size: {len(dataset['train'])} | Validation dataset size: {len(dataset['test'])}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "validation_songs.txt"), "w") as f:
        f.write("Songs used for validation:\n")
        f.write("-" * 40 + "\n")
        for item in dataset['test']:
            audio_url = item['messages'][0]['content'][0]['audio_url']
            song_name = audio_url.split("/")[-1].replace("_bpm.wav", "")
            f.write(f"{song_name}\n")

    def preprocess_function(examples):
        input_ids = []
        attention_mask = []
        labels = []
        audio_features = []
        feature_attention_mask = []
        inputs = {}
        
        for messages in examples['messages']:
            user_msg = messages[0]
            audio_url = user_msg['content'][0]['audio_url']
            
            if "/Users/mukeshguntumadugu/" in audio_url:
                audio_url = audio_url.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")
            
            y, sr = librosa.load(audio_url, sr=processor.feature_extractor.sampling_rate)
            
            user_text = messages[0]['content'][1]['text']
            assistant_text = messages[1]['content'][0]['text']
            
            text = (
                "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n"
                f"<|im_start|>user\\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\\n{user_text}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{assistant_text}<|im_end|>\\n"
            )
            
            inputs = processor(
                text=text,
                audio=[y],
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt"
            )
            
            label = inputs["input_ids"].clone()
            
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
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    print("Injecting LoRA adapters...")
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
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="no", # Save only at the very end to conserve space
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        learning_rate=3e-4, # slightly higher learning rate for a simple task
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
        data_collator=MultimodalDataCollator(processor),
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving final model adapter...")
    trainer.save_model(OUTPUT_DIR)
    
    # Extract Loss
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

    print(f"Done. Checkpoints and loss CSVs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
