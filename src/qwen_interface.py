
import os
import torch
import librosa
import soundfile as sf
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from io import BytesIO

# Default Model Path (can be overriden or local path)
DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

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
        max_new_tokens=4096,         # Dropped to 4096 so it finishes reasonably fast even on large outputs
        do_sample=True,              # Changed to True with low temperature so it doesn't loop infinitely
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )
    
    if _prefix_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = _prefix_fn
        
    with torch.no_grad():
        generated_ids = _model.generate(**generate_kwargs)
        
    # Decode
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response_text = _processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response_text

if __name__ == "__main__":
    # Test Block
    TEST_AUDIO = "test.ogg" # Replace with real path if testing
    if os.path.exists(TEST_AUDIO):
        print(generate_beatmap_with_qwen(TEST_AUDIO, "Generate a beatmap."))
