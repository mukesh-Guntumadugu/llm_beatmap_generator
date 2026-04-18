import os
import argparse
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

# Suppress annoying warnings
warnings.filterwarnings('ignore')

import music_flamingo_interface

def plot_for_audio(file_path, output_dir):
    print(f"\n========================================")
    print(f"PROBING FLAMINGO: {os.path.basename(file_path)}")
    print(f"========================================")
    
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("1. Extracting mathematical ground truths (Librosa)...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    print(f"   -> Detected {len(beat_times)} structural beats. (BPM: {tempo[0] if isinstance(tempo, np.ndarray) else tempo:.2f})")
    
    model = music_flamingo_interface._model
    processor = music_flamingo_interface._processor
    
    # Flamingo specifically relies heavily on proper 16000Hz internal loading
    y_flam, _ = librosa.load(file_path, sr=16000)
    
    messages = [{"role": "user", "content": [{"type": "audio", "audio_url": "dummy"}, {"type": "text", "text": "probe"}]}]
    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(text=formatted_prompt, audio=[y_flam], return_tensors="pt", sampling_rate=16000)
    # Move to GPU safely
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    # Specifically cast floating inputs (audio) to float16 to match Flamingo's loaded dtype
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.float16)
    
    print("2. Pushing audio through Flamingo's AudioMAE backbone...")
    features = None
    with torch.no_grad():
        try:
            hidden_states = None
            
            # ATTEMPT 1: Dynamic extraction of the audio encoder directly
            audio_encoder = None
            if hasattr(model, 'model') and hasattr(model.model, 'audio_encoder'):
                audio_encoder = model.model.audio_encoder
            elif hasattr(model, 'audio_encoder'):
                audio_encoder = model.audio_encoder
            elif hasattr(model, 'audio_tower'):
                audio_tower = model.audio_tower
                
            if audio_encoder:
                audio_inputs = inputs.get('input_features', inputs.get('audio_features', None))
                if audio_inputs is not None:
                    audio_outputs = audio_encoder(audio_inputs, output_hidden_states=True)
                    hidden_states = audio_outputs.hidden_states[-1]
            
            # ATTEMPT 2: If Audio encoder was inaccessible, push through the entire LLM and extract the deepest contextualized hidden states
            if hidden_states is None:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states:
                    hidden_states = outputs.encoder_hidden_states[-1]
                else:
                    hidden_states = outputs.hidden_states[-1]
                    
            if hidden_states is None:
                raise ValueError("Could not extract any hidden states.")
            
            features = hidden_states[0].cpu().numpy()
            
        except Exception as e:
            print(f"   [!] Flamingo Neural Extraction Failed: {e}")
            return
            
    print(f"   -> Extracted latent tokens with dimensions: {features.shape}.")
    
    print("3. Compressing Flamingo latents via PCA...")
    pca = PCA(n_components=1)
    
    # Flamingo's token dimensions might be (seq_len, dim) or flat. Attempt auto-reshape.
    if len(features.shape) > 2:
        features = features.reshape(-1, features.shape[-1])
        
    flam_1d = pca.fit_transform(features).flatten()
    
    if abs(min(flam_1d)) > abs(max(flam_1d)):
        flam_1d = -flam_1d
        
    flam_1d = (flam_1d - flam_1d.min()) / (flam_1d.max() - flam_1d.min() + 1e-8)
    onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-8)
    
    flam_times = np.linspace(0, duration, len(flam_1d))
    
    print("4. Generating visual correlation chart...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_flamingo_onset_probe.png")
    
    plt.figure(figsize=(18, 6))
    plt.plot(times, onset_norm, label="Librosa Mathematical Onset", color="blue", alpha=0.35, linewidth=2)
    for i, b in enumerate(beat_times):
        plt.axvline(x=b, color='green', alpha=0.3, linestyle='--', linewidth=1, label="Mathematical Drum Beats" if i == 0 else "")
        
    plt.plot(flam_times, flam_1d, label="Flamingo Latent Activation (1D PCA)", color="purple", linewidth=2.5)
    
    plt.title(f"Music-Flamingo Understanding vs Physical Tempo\nTarget: {base_name}", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=11)
    plt.ylabel("Activation Spike Intensity (Normalized)", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    
    print(f"✅ Flamingo Chart exported: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, help="Directory containing audio")
    parser.add_argument('--output_dir', type=str, default="results_flamingo_probe_fraxtil", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n--- Initializing Music-Flamingo Audio Engine ---")
    music_flamingo_interface.setup_music_flamingo()
    
    supported_exts = ['.ogg', '.mp3', '.wav']
    files = []
    for root, _, filenames in os.walk(args.target_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in supported_exts:
                files.append(os.path.join(root, f))
                
    if not files:
        print(f"❌ No audio files found.")
        return
        
    for fp in files:
        plot_for_audio(fp, args.output_dir)

if __name__ == "__main__":
    main()
