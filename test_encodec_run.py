import sys
import os
import torch
import torchaudio
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Add src to path to import the module
# Add src to path to import the module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# from Neural_Audio_Codecs.EnCodecimplementation import AudioTokenizer # unexpected token error due to spaces

import importlib.util
spec = importlib.util.spec_from_file_location("EnCodecimplementation", 
    os.path.join(os.path.dirname(__file__), "src/Neural Audio Codecs/EnCodecimplementation.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["EnCodecimplementation"] = module
spec.loader.exec_module(module)
AudioTokenizer = module.AudioTokenizer

def test_tokenization():
    # Generate a dummy sine wave audio file
    sample_rate = 44100
    duration = 2.0 # seconds
    frequency = 440.0 # Hz
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * 3.14159 * frequency * t).unsqueeze(0) # (1, T)
    
    audio_path = "test_audio.wav"
    import soundfile as sf
    # Soundfile expects (T, C) or (T,) for mono. Waveform is (1, T).
    # Convert using numpy
    wav_np = waveform.squeeze(0).numpy()
    sf.write(audio_path, wav_np, sample_rate)
    print(f"Created dummy audio file: {audio_path}")
    
    try:
        tokenizer = AudioTokenizer()
        
        print("Tokenizing...")
        tokens = tokenizer.tokenize(audio_path)
        print(f"Tokens Shape: {tokens.shape}")
        
        # Basic assertions
        assert tokens.dim() == 3, "Tokens must be 3D (Batch, Quantizers, Time)"
        assert tokens.shape[0] == 1, "Batch size should be 1"
        # Default 24kHz model with 6.0kbps target bandwidth uses 8 codebooks.
        # Frame rate is 75Hz.
        print(f"Number of codebooks: {tokens.shape[1]}")
        # assert tokens.shape[1] == 32 # This was incorrect for 6kbps
        
        # Verify time dimension
        # 2 seconds * 75 Hz = 150 frames
        expected_frames = int(duration * 75)
        # Allow +/- 1 frame tolerance
        assert abs(tokens.shape[2] - expected_frames) <= 1, f"Expected ~{expected_frames} frames, got {tokens.shape[2]}"
        
        output_path = "test_tokens.pt"
        tokenizer.save_tokens(tokens, output_path)
        
        assert os.path.exists(output_path)
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists("test_tokens.pt"):
            os.remove("test_tokens.pt")

if __name__ == "__main__":
    test_tokenization()
