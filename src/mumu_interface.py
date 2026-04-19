"""
mumu_interface.py
=================
Loads the MuMu-LLaMA model and performs inference.

NOTE: This requires the MuMu-LLaMA repository to be cloned and the LLaMA-2 
weights to be manually downloaded to its ckpts/ directory.
"""

import sys
import os

# Placeholder for MuMu-LLaMA model and processor
from src.mumu_measure_interface import initialize_mumu_model
import torch
import torchaudio
import traceback

# State initialized by mumu_measure_interface
_is_initialized = False

def setup_mumu():
    """Initializes the MuMu-LLaMA model."""
    global _is_initialized
    if not _is_initialized:
        print("Loading MuMu-LLaMA model (Actual integration)...")
        initialize_mumu_model()
        _is_initialized = True
        print("✅ MuMu-LLaMA real interface initialized.")

def generate_beatmap_with_mumu(audio_path: str, prompt: str) -> str:
    """Passes the audio file and prompt to the MuMu-LLaMA model for strict generation."""
    global _is_initialized
    if not _is_initialized:
        setup_mumu()
        
    print(f"Generating with MuMu-LLaMA for: {os.path.basename(audio_path)}")
    
    try:
        model, tokenizer = initialize_mumu_model()
        import llama
        
        waveform, sr = torchaudio.load(audio_path)
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
        
        # MuMu's encode_audio feeds this directly to MERT's AutoProcessor which expects
        # a plain 1D numpy float32 array at 24kHz.  Passing a torch tensor adds a
        # dimension each time MuMu wraps it in [[audios], 1], causing a [1,1,1,N] crash.
        audio_np = torch.mean(waveform, 0).float().cpu().numpy()  # shape: (N,) float32
        
        formatted_prompt = llama.utils.format_prompt(prompt)
        
        with torch.no_grad():
            # model is already float32 on GPU; no autocast needed
            results = model.generate(
                prompts=[formatted_prompt],
                audios=audio_np,         # 1D numpy array — MERT processor handles batching
                max_gen_len=1024,
                temperature=0.1,
                top_p=0.9
            )
                
        if isinstance(results, list) and len(results) > 0:
            return str(results[0])
        return str(results)
        
    except Exception as e:
        print(f"❌ Error during actual MuMu inference: {e}")
        traceback.print_exc()
        return ""
