
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

def setup_qwen(model_id=DEFAULT_MODEL_ID, device="cuda"):
    """
    Loads the Qwen2-Audio model and processor.
    For a cluster, assume 'cuda' is available.
    """
    global _model, _processor
    
    print(f"Loading Qwen2-Audio model: {model_id}...")
    try:
        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        _model.eval()
        print("✅ Qwen2-Audio loaded successfully.")
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
    # Let's load it to ensure correct sampling rate.
    
    # Note: Qwen2-Audio usually expects 16kHz? Processor handles resampling if we pass array + sampling_rate?
    # HuggingFace documentation says input can be file path or array.
    # Let's try passing the waveform array.
    
    y, sr = librosa.load(audio_path, sr=_processor.feature_extractor.sampling_rate)
    
    # Prepare Conversation using standard Chat Template
    # Qwen2-Audio-Instruct expects a specific format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply template
    # Note: Qwen2 processor handles audio token insertion if we use apply_chat_template correctly
    text = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Load and process Audio
    audios = []
    # If using local file, we load it. librosa ensures sampling rate matches
    y, sr = librosa.load(audio_path, sr=_processor.feature_extractor.sampling_rate)
    audios.append(y)
    
    inputs = _processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = inputs.to(_model.device)

    # Generate
    print("Generating with Qwen2-Audio...")
    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=4096)
        
    # Decode
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response_text = _processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response_text

if __name__ == "__main__":
    # Test Block
    TEST_AUDIO = "test.ogg" # Replace with real path if testing
    if os.path.exists(TEST_AUDIO):
        print(generate_beatmap_with_qwen(TEST_AUDIO, "Generate a beatmap."))
