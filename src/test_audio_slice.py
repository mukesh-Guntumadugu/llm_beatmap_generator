import librosa
import soundfile as sf
import os
import numpy as np

# Path to the file
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_dir, "musicForBeatmap", "Goin' Under", "Goin' Under.ogg")
output_path = os.path.join(script_dir, "test_slice.ogg")

print(f"Loading {audio_path}...")
try:
    # Load 15 seconds just to be safe, we want 10s
    y, sr = librosa.load(audio_path, sr=None, duration=15.0)
    print(f"Loaded successfully. Rate: {sr}, Shape: {y.shape}")
    
    # Slice exactly 10 seconds
    # num_samples = 10 * sr
    # y_slice = y[:num_samples]
    
    # Actually librosa.load with duration already slices it, but let's be explicit if we loaded full
    # Since we used duration=15, let's slice 10s
    y_slice = y[:10*sr]
    
    print(f"Saving 10s slice to {output_path}...")
    sf.write(output_path, y_slice, sr)
    print("Success.")
    
except Exception as e:
    print(f"Error: {e}")
