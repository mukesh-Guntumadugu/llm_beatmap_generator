
import os
import torch
import librosa
import soundfile as sf
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from io import BytesIO

# Default Model Path (can be overriden or local path)
DEFAULT_MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"

_model = None
_processor = None
_prefix_fn = None

def setup_qwen(model_id=DEFAULT_MODEL_ID, device=None):
    """
    Loads the Qwen2-Audio model and processor.
    Auto-detects CUDA availability -- uses float16 on GPU, float32 on CPU.
    """
    global _model, _processor

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("⚠️  WARNING: No GPU detected. Running Qwen on CPU will be extremely slow.")
        print("   Submit a SLURM GPU job instead:  sbatch run_qwen.sh")
        raise RuntimeError("No CUDA GPU available. Refusing to run on CPU for performance reasons.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading Qwen2-Audio model: {model_id}...")
    print(f"  Device: {device} | dtype: {dtype}")
    try:
        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else "cpu",
            trust_remote_code=True,
            torch_dtype=dtype
        )
        try:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, "/data/mg546924/models/qwen2-audio-lora-onsets/checkpoint-2208")
            print("✅ 15-Hour Trained LoRA Adapter (checkpoint-2208) successfully loaded!")
        except Exception as e:
            print(f"⚠️ Could not attach trained weights (LoRA): {e}")
        
        _model.eval()

        # ── Set up lm-format-enforcer constrained decoding ───────────────────────
        try:
            from lmformatenforcer import RegexParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn
            )
            
            # Regex for CSV rows: time_ms,beat_pos,notes,placement,note_type,conf,instrument
            # Allows notes like '1000' or '","' for separators
            csv_row = r"\d+\.\d+,\d+\.\d+,(?:[0123M]{4}|\",\"),-?\d+,-?\d+,\d+\.\d+,[a-z]+"
            csv_regex = f"({csv_row}\n)+"
            schema_parser = RegexParser(csv_regex)
            
            global _prefix_fn
            _prefix_fn = build_transformers_prefix_allowed_tokens_fn(
                _processor.tokenizer, schema_parser
            )
            print("✅ lm-format-enforcer constrained decoding enabled (Regex: CSV Rows).")
        except ImportError:
            _prefix_fn = None
            print("⚠️  lm-format-enforcer not installed — falling back to unconstrained generation.")
            print("   To enable: pip install lm-format-enforcer")
        except Exception as e:
            _prefix_fn = None
            print(f"⚠️  Constrained decoding setup failed: {e} — using unconstrained generation.")

        print(f"✅ Qwen2-Audio loaded successfully on {device}.")
    except Exception as e:
        print(f"❌ Failed to load Qwen2-Audio: {e}")
        raise

def generate_beatmap_with_qwen(audio_path: str, prompt: str) -> str:
    """
    Generates beatmap text using Qwen2-Audio.
    """
    global _model, _processor
    
    if not _model or not _processor:
        setup_qwen()

    # Preprocess Audio
    # Qwen2Audio expects raw audio data or path?
    # The processor can handle audio arrays.
    target_sr = _processor.feature_extractor.sampling_rate
    y, sr = librosa.load(audio_path, sr=target_sr)

    # Prepare Conversation using standard Chat Template
    # Qwen2-Audio-Instruct expects a file:// URI for local audio files.
    # A bare file path causes the processor to silently skip audio embedding,
    # making the model respond: "I cannot access audio files."
    audio_uri = f"file://{os.path.abspath(audio_path)}"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_uri},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply template
    text = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Pass audio with explicit sampling_rate to suppress the warning
    inputs = _processor(text=text, audio=[y], sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = inputs.to(_model.device)

    # Generate
    print("Generating with Qwen2-Audio...")
    
    generate_kwargs = dict(
        **inputs, 
        max_new_tokens=8192,
        do_sample=False,
        repetition_penalty=1.0
    )
    
    if _prefix_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = _prefix_fn
        
    with torch.no_grad():
        generated_ids = _model.generate(**generate_kwargs)
        
    # Decode
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response_text = _processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response_text

def get_qwen_16_step_probabilities(
    audio_path: str, prompt: str, candidates: list,
    temperature=5.0, top_p=0.9, min_p=0.05, top_k=None,
    repetition_penalty=1.2, recent_history=None
) -> dict:
    """ Computes exact sequence generation probabilities for specific text candidates by performing forward-pass loss extraction. """
    global _model, _processor
    if not _model or not _processor:
        setup_qwen()

    target_sr = _processor.feature_extractor.sampling_rate
    y, sr = librosa.load(audio_path, sr=target_sr)
    audio_uri = f"file://{os.path.abspath(audio_path)}"
    
    conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_uri}, {"type": "text", "text": prompt}]}]
    text_context = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Process base context to find token length
    base_inputs = _processor(text=text_context, audio=[y], sampling_rate=target_sr, return_tensors="pt", padding=True).to(_model.device)
    base_input_len = base_inputs.input_ids.shape[1]
    
    cand_scores = []
    
    for cand in candidates:
        cand_text = text_context + cand
        inputs = _processor(text=cand_text, audio=[y], sampling_rate=target_sr, return_tensors="pt", padding=True).to(_model.device)
        
        labels = inputs.input_ids.clone()
        labels[0, :base_input_len] = -100 # Mask out everything except the candidate tokens
        
        with torch.no_grad():
            outputs = _model(**inputs, labels=labels)
            loss = outputs.loss 
            # CrossEntropy is implicitly averaged over the non-masked (candidate) tokens
            num_cand_tokens = (labels != -100).sum().item()
            seq_log_prob = -loss.item() * num_cand_tokens
            cand_scores.append(seq_log_prob)
            
    scores_tensor = torch.tensor(cand_scores, dtype=torch.float32)

    # 1. Dynamic Repetition Penalty 
    # (If a step was recently picked, multiply its negative NLL by >1 to push probability toward 0)
    if recent_history and repetition_penalty > 1.0:
        for i, cand in enumerate(candidates):
            if cand == "0000":
                continue  # Never penalize resting!
                
            count = recent_history.count(cand)
            if count > 0:
                # Scaled Punishment: e.g. Penalty 1.15 -> 1 repeat = 1.15x, 3 repeats = 1.45x
                dyn_penalty = 1.0 + (count * (repetition_penalty - 1.0))
                
                # Soft slap on wrist for immediate duplicate double-tapping (the exact b-1 beat)
                if cand == recent_history[-1]:
                    dyn_penalty *= 1.1
                    
                scores_tensor[i] = scores_tensor[i] * dyn_penalty

    # 2. Temperature Scaling
    if temperature != 1.0 and temperature > 0.0:
        scores_tensor = scores_tensor / temperature
        
    # Baseline Softmax conversion
    probs = torch.nn.functional.softmax(scores_tensor, dim=0)

    # 3. Min-P Filter
    if min_p > 0.0:
        max_prob = torch.max(probs)
        min_allowed = max_prob * min_p
        probs[probs < min_allowed] = 0.0

    # 4. Top-K Filter (Explicit Array Shaving)
    if top_k is not None and top_k < len(probs):
        top_k_values, _ = torch.topk(probs, k=top_k)
        if len(top_k_values) > 0:
            kth_best_prob = top_k_values[-1]
            probs[probs < kth_best_prob] = 0.0

    # 5. Top-P (Nucleus) Filter
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Identify indices to zero out (cumulative sum > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right by 1 to include the token that formally tips over the top_p boundary
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        
    # Re-normalize remaining valid percentages perfectly back to 100%
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
        
    probs = probs.numpy()
    
    return {candidates[i]: probs[i]*100 for i in range(len(candidates))}

if __name__ == "__main__":
    # Test Block
    TEST_AUDIO = "test.ogg" # Replace with real path if testing
    if os.path.exists(TEST_AUDIO):
        print(generate_beatmap_with_qwen(TEST_AUDIO, "Generate a beatmap."))
