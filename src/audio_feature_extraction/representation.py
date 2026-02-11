import librosa
import numpy as np
import sys
import os

class AudioRepresentation:
    def __init__(self, sample_rate=44100, hop_length_ms=10, n_mels=80):
        self.sample_rate = sample_rate
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.n_mels = n_mels
        # Window sizes in ms: 5, 10, 23, 46, 93, 186
        self.window_sizes_ms = [5, 10, 23, 46, 93, 186]
        self.n_ffts = [int(sample_rate * ms / 1000) for ms in self.window_sizes_ms]
        
        # Ensure n_ffts are powers of 2 for efficiency (optional but good practice, 
        # though librosa handles non-powers of 2 fine. We'll use calculated values directly 
        # to match the ms requirements exactly: ~1014, ~2028, ~4101 samples)
        # Actually standard practice is usually finding nearest power of 2 or just using the calculated window size.
        # Given "window lengths of 23ms...", we will use the calculated lengths as the window size.

    def load_audio(self, audio_path):
        try:
            print(f"Loading {audio_path} at {self.sample_rate}Hz...")
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            print(f"Audio loaded successfully. Shape: {y.shape}")
            return y
        except Exception as e:
            print(f"Error loading audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_file(self, audio_path):
        print("Calling load_audio...")
        y = self.load_audio(audio_path)
        if y is None:
            print("load_audio returned None")
            return None
        if len(y) == 0:
            print("load_audio returned empty array")
            return None
        print(f"Processing audio with {len(y)} samples...")

        features_list = []

        # Compute Spectrograms for each window size
        for n_fft in self.n_ffts:
            # STFT
            # We use the same hop length for all to ensure they align in time
            S = librosa.stft(y, n_fft=n_fft, hop_length=self.hop_length, window='hann')
            
            # Magnitude
            S_mag = np.abs(S)**2 # Power spectrogram for mel filter

            # Mel Filterbank
            mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=n_fft, n_mels=self.n_mels)
            S_mel = np.dot(mel_basis, S_mag)

            # Log Scaling
            # log(1 + S) matches the description closer than simple log10, or log10(1 + 10 * S) often used.
            # "scale the filter outputs logarithmically" - typically librosa.power_to_db or log1p.
            # Schluter & Bock 2014 use logarithmic magnitudes. 
            # We'll use log(1 + C * S) formulation or just standard Log-Mel. 
            # librosa.power_to_db is standard: 10 * log10(S / ref). 
            # Let's use log(1 + S) for simplicity and stability as implied by "logarithmically".
            S_log = np.log1p(S_mel)

            features_list.append(S_log)

        # Stack features: (n_mels, time, 3)
        # Ensure they all have same time dimension (they should due to fixed hop_length)
        min_time = min([f.shape[1] for f in features_list])
        features_list = [f[:, :min_time] for f in features_list]
        
        # Shape: (80, Time, 3)
        stacked_features = np.stack(features_list, axis=-1)
        
        # Transpose to (Time, 80, 3) for easier windowing
        # Shape: (Time, 80, 3)
        stacked_features = stacked_features.transpose(1, 0, 2)

        # Normalize (Zero Mean, Unit Variance)
        # "normalize each frequency band to zero mean and unit variance"
        # We compute mean/std along time axis.
        mean = np.mean(stacked_features, axis=0, keepdims=True)
        std = np.std(stacked_features, axis=0, keepdims=True)
        
        # Avoid divide by zero
        std[std == 0] = 1.0
        
        normalized_features = (stacked_features - mean) / std

        # Context Windowing
        # "prepend and append seven frames of past and future context"
        # Context size 15 means +/- 7 frames.
        pad_width = 7
        
        # Pad the time axis (axis 0)
        # ((pad_before, pad_after), (pad_freq, pad_freq), (pad_channel, pad_channel))
        padded_features = np.pad(normalized_features, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
        
        # Window
        num_frames = normalized_features.shape[0]
        # Result shape: (Num_Frames, 15, 80, 3)
        
        # Efficient sliding window with numpy strides
        # For simplicity in this script, a loop or stride_tricks can be used.
        # Given we want to return a tensor, let's use a simple list comp or creating a view.
        
        # output_tensor = []
        # for i in range(num_frames):
        #     window = padded_features[i : i + 15]
        #     output_tensor.append(window)
        # return np.array(output_tensor)
        
        # Memory efficient view:
        from numpy.lib.stride_tricks import as_strided
        
        # Current shape: (Time_Padded, 80, 3)
        # Desired: (Time, 15, 80, 3)
        
        window_size = 15
        
        shape = (num_frames, window_size, 80, 6) if len(self.window_sizes_ms) == 6 else (num_frames, window_size, 80, len(self.window_sizes_ms))
        strides = (padded_features.strides[0], padded_features.strides[0], padded_features.strides[1], padded_features.strides[2])
        
        rolling_view = as_strided(padded_features, shape=shape, strides=strides)
        return rolling_view
        
    def visualize_and_save(self, features, output_path="audio_representation.npy"):
        # Save to .npy
        np.save(output_path, features)
        print(f"Tensor saved to: {output_path}")
        
        # Visualize
        # features shape: (Num_Frames, 15, 80, N_Scales)
        # To visualize, we want to look at the 'center' of the window for each frame, 
        # effectively reconstructing the (Num_Frames, 80, N_Scales) spectrograms.
        # The center of the 15-frame window is index 7.
        
        # Extract the center frame for all time steps
        # Shape: (Num_Frames, 80, N_Scales)
        spectrograms = features[:, 7, :, :]
        
        # Transpose for plotting: (N_Scales, 80, Num_Frames) -> (Channels, Freq, Time)
        spectrograms = spectrograms.transpose(2, 1, 0)
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # timestamps
        timestamps = self.window_sizes_ms
        n_scales = len(timestamps)
        hop_time_ms = 10 # 10ms hop
        
        # X-axis array in milliseconds
        num_frames = spectrograms.shape[2]
        x_ms = np.arange(num_frames) * hop_time_ms
        
        # Create subplots
        fig = make_subplots(rows=n_scales, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.02,
                            subplot_titles=[f"Window: {ms}ms" for ms in timestamps])
        
        for i in range(n_scales):
            # Spectrogram[i] shape: (80, Time)
            # Plotly Heatmap expects z as (y, x) or similar, but let's be explicit
            # x: time labels, y: frequency bands (0-79)
            fig.add_trace(
                go.Heatmap(
                    z=spectrograms[i],
                    x=x_ms,
                    # y=np.arange(80), 
                    colorscale='Jet', # High contrast
                    name=f'{timestamps[i]}ms',
                    showscale=(i==n_scales-1) # Show scale only once (maybe at bottom or right)
                ),
                row=i+1, col=1
            )
            
            fig.update_yaxes(title_text=f"{timestamps[i]}ms", row=i+1, col=1)
            
        fig.update_xaxes(title_text="Time (ms)", row=n_scales, col=1)
        
        fig.update_layout(
            title_text=f"Multi-Scale Audio Representation (Total: {num_frames * hop_time_ms} ms)",
            # Increase height to accommodate more plots
            height=300 * n_scales,
            dragmode='zoom', # Enable zoom
            hovermode='x unified' # Show values for all plots at x
        )
        
        # Save as HTML
        html_path = output_path.replace(".npy", ".html")
        fig.write_html(html_path)
        print(f"Interactive visualization saved to: {html_path}")
        
        # Try to open it
        try:
            import webbrowser
            # Convert to absolute path for browser
            abs_path = "file://" + os.path.abspath(html_path)
            webbrowser.open(abs_path)
        except:
            print(f"Could not open browser automatically. Please open {html_path} manually.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python representation.py <audio_file> [output_npy_path]")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    output_path = "audio_representation.npy"
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        sys.exit(1)

    print(f"Processing: {audio_file}")
    
    rep = AudioRepresentation()
    features = rep.process_file(audio_file)
    
    if features is not None:
        print(f"Success! Output Tensor Shape: {features.shape}")
        # Expected: (Num_Original_Frames, 15, 80, 3)
        print(f" - Num Frames: {features.shape[0]}")
        print(f" - Context Window: {features.shape[1]} (Expected 15)")
        print(f" - Frequency Bands: {features.shape[2]} (Expected 80)")
        print(f" - Channels (Scales): {features.shape[3]} (Expected 3)")
        
        # Sample check
        print(f"Sample data mean: {np.mean(features):.4f}")
        print(f"Sample data std: {np.std(features):.4f}")
        
        # Visualize and Save
        rep.visualize_and_save(features, output_path)
    else:
        print("Failed to process audio.")
