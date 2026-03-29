import os
import torch
import librosa
import traceback

_processor = None
_model = None

# Set HF HOME strictly to the data directory before loading models
os.environ['HF_HOME'] = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints"

def setup_music_flamingo():
    """
    Initializes the Music Flamingo model using HuggingFace Transformers.
    Downloads weights to the /data pipeline cache on first run.
    """
    global _processor, _model
    
    try:
        from transformers import AudioFlamingo3Processor, AudioFlamingo3ForConditionalGeneration
    except ImportError as e:
        print(f"❌ Failed to import Music Flamingo classes. Ensure you are using the music_flamingo_env environment: {e}")
        raise
        
    print("Loading Music Flamingo Processor...")
    _processor = AudioFlamingo3Processor.from_pretrained("nvidia/music-flamingo-hf", trust_remote_code=True)
    
    print("Loading Music Flamingo Model (this loads ~30GB into GPU)...")
    _model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        "nvidia/music-flamingo-hf", 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    _model.eval()
    
    print("✅ Music Flamingo interface initialized.")

def generate_beatmap_with_flamingo(audio_path: str, prompt: str) -> str:
    """
    Passes the audio file and prompt to the Music Flamingo model.
    """
    global _processor, _model
    if not _model or not _processor:
        setup_music_flamingo()
        
    try:
        # Load audio at native 16000Hz preferred sample rate
        y, sr = librosa.load(audio_path, sr=16000)
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "dummy"},
                {"type": "text", "text": prompt}
            ]}
        ]
        formatted_prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = _processor(text=formatted_prompt, audio=[y], return_tensors="pt", sampling_rate=sr)
        inputs = {k: v.to(_model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.float16)
        
        with torch.inference_mode():
            generated_ids = _model.generate(
                **inputs, 
                max_new_tokens=8192,
                do_sample=False,
                temperature=0.01,
                repetition_penalty=1.0
            )
            
        # Decode the output tokens back to text string
        response = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Strip out the echoed prompt context if the model repeated it
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
            
        return response
        
    except Exception as e:
        traceback.print_exc()
        print("Error during Music Flamingo generation:", e)
        return ""
