import os
import argparse
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

# Suppress annoying librosa/pytorch warnings for cleaner CLI
warnings.filterwarnings('ignore')

import qwen_interface

def plot_for_audio(file_path, output_dir):
    print(f"\n========================================")
    print(f"PROBING: {os.path.basename(file_path)}")
    print(f"========================================")
    
    # 1. Librosa Ground Truth extraction
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("1. Extracting mathematical ground truths (Librosa)...")
    # Extract Onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    
    # Extract Tempo/Beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    print(f"   -> Detected {len(beat_times)} structural beats.")
    print(f"   -> Estimated BPM: {tempo[0] if isinstance(tempo, np.ndarray) else tempo:.2f}")
    
    # 2. Qwen Forward Pass (Audio Extraction only)
    model = qwen_interface._model
    processor = qwen_interface._processor
    
    # Needs to match Qwen's specific expected audio sample rate (usually 16000Hz)
    target_sr = processor.feature_extractor.sampling_rate
    y_qwen, _ = librosa.load(file_path, sr=target_sr)
    
    # Qwen2Audio processor explicitly requires text formatting even if we only want audio embeddings
    audio_uri = f"file://{os.path.abspath(file_path)}"
    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio_url": audio_uri}, {"type": "text", "text": "probe"}]}
    ]
    text_context = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(text=text_context, audio=[y_qwen], sampling_rate=target_sr, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    print("2. Pushing audio through Qwen's neural audio_tower...")
    with torch.no_grad():
        # Attempt to dynamically locate the Whisper audio tower inside the model architecture
        audio_tower = None
        if hasattr(model, 'model') and hasattr(model.model, 'audio_tower'):
            audio_tower = model.model.audio_tower
        elif hasattr(model, 'audio_tower'):
            audio_tower = model.audio_tower
            
        try:
            if audio_tower:
                # Retrieve pure, unadulterated hidden states before LLM text projection
                audio_outputs = audio_tower(inputs.input_features.to(model.device), output_hidden_states=True)
                hidden_states = audio_outputs.hidden_states[-1] # Shape: [batch, sequence_length, hidden_dimensions]
            else:
                raise ValueError("Could not dynamically locate 'audio_tower' in Qwen architecture via introspection.")
                
            features = hidden_states[0].cpu().numpy() # Extract [seq_len, dim]
            
        except Exception as e:
            print(f"   [!] Neural Extraction Failed: {e}")
            return
            
    print(f"   -> Extracted {features.shape[0]} latent tokens, each with {features.shape[1]} dimensions.")
    
    # 3. PCA Compression
    print("3. Compressing Qwen's high-dimensional latent space to 1D via PCA...")
    pca = PCA(n_components=1)
    qwen_1d = pca.fit_transform(features).flatten()
    
    # If the PCA principal component is flipped (negative spikes for transient noise), invert it for visuals
    if abs(min(qwen_1d)) > abs(max(qwen_1d)):
        qwen_1d = -qwen_1d
        
    # Scale both between 0 and 1 so they overlay beautifully on the graph
    qwen_1d = (qwen_1d - qwen_1d.min()) / (qwen_1d.max() - qwen_1d.min() + 1e-8)
    onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-8)
    
    # Qwen's token sequence length doesn't precisely match standard sample rates. Let's interpolate it across real-time length.
    qwen_times = np.linspace(0, duration, len(qwen_1d))
    
    # 4. Plotting Data Validation
    print("4. Generating visual correlation chart...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_qwen_onset_probe.png")
    
    plt.figure(figsize=(18, 6))
    
    # A. Mathematical Ground Truth
    plt.plot(times, onset_norm, label="Librosa Mathematical Onset", color="blue", alpha=0.35, linewidth=2)
    for i, b in enumerate(beat_times):
        plt.axvline(x=b, color='green', alpha=0.3, linestyle='--', linewidth=1, label="Mathematical Drum Beats" if i == 0 else "")
        
    # B. The AI (Black Box)
    plt.plot(qwen_times, qwen_1d, label="Qwen Latent Activation (1D Token Space)", color="red", linewidth=2.5)
    
    plt.title(f"Qwen Token Understanding vs Physical Tempo\nTarget: {base_name}", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=11)
    plt.ylabel("Activation Spike Intensity (Normalized)", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(out_file, dpi=180)
    plt.close()
    
    print(f"✅ Chart exported beautifully to: {out_file}!")

def main():
    parser = argparse.ArgumentParser(description="Probe Qwen Audio tokens to prove spatial understanding.")
    parser.add_argument('--target_dir', type=str, required=True, help="Directory containing audio files (.ogg/.mp3/.wav)")
    parser.add_argument('--output_dir', type=str, default="results_qwen_probe", help="Output directory for generated PNG charts.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Engage Qwen LLM using the existing cluster configurations
    print("\n--- Initializing Qwen Audio Engine ---")
    qwen_interface.setup_qwen()
    
    # Collect targets
    supported_exts = ['.ogg', '.mp3', '.wav']
    files = []
    for root, _, filenames in os.walk(args.target_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in supported_exts:
                files.append(os.path.join(root, f))
    
    if not files:
        print(f"❌ No valid audio files found in {args.target_dir}.")
        return
        
    print(f"\nFound {len(files)} target files to process!")
    for fp in files:
        plot_for_audio(fp, args.output_dir)
        
    print(f"\n--- Batch Pipeline Complete! ---")
    print(f"You can now download the `.png` charts from '{args.output_dir}' off your cluster.")

if __name__ == "__main__":
    main()
