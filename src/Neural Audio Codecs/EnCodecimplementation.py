import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import typing as tp
import os
import ssl

# Fix for SSL certificate verification failure on some systems
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class AudioTokenizer:
    """
    A class to tokenize audio using Meta's EnCodec model.

     to run:    python3 "src/Neural Audio Codecs/EnCodecimplementation.py" "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg"


    """
    def __init__(self, device: str = 'cpu', target_bandwidth: float = 6.0):
        """
        Initialize the AudioTokenizer.
        
        Args:
            device: 'cpu' or 'cuda'
            target_bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
                              Higher bandwidth = better quality = more tokens/info.
                              Default 6.0 is a good balance.
        """
        self.device = device
        # Use the 24kHz model which is standard for music/general audio
        print(f"Loading EnCodec model (24kHz) on {device}...")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(target_bandwidth)
        self.model.to(device)
        self.model.eval()
        print("Model loaded.")

    def tokenize(self, audio_path: str) -> torch.Tensor:
        """
        Tokenize an audio file into discrete codes.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            torch.Tensor: Tensor of shape (Batch, Quantizers, TimeSteps) containing the codes.
                          Batch is usually 1.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        import soundfile as sf
        # Load audio using soundfile to avoid torchaudio backend issues
        wav_np, sr = sf.read(audio_path)
        
        # Convert to tensor
        # soundfile returns (Time, Channels) or (Time,)
        wav = torch.from_numpy(wav_np).float()
        
        if wav.dim() == 1:
            # Mono (Time,) -> (1, Time)
            wav = wav.unsqueeze(0)
        else:
            # (Time, Channels) -> (Channels, Time)
            wav = wav.t()
            
        # torchaudio.load returns (C, T), so we are good.
        
        # EnCodec expects input shape (Batch, Channels, Time)
        wav = wav.unsqueeze(0)
        
        
        if self.model.channels == 1 and wav.shape[1] > 1:
            # print("Downmixing to mono...")
            wav = wav.mean(dim=1, keepdim=True)
            
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        
        wav = wav.to(self.device)

        # Extract discrete codes
        with torch.no_grad():
            # encode returns a list of (codes, scale) tuples
            # codes shape: (Batch, NumQuantizers, TimeSteps)
            encoded_frames = self.model.encode(wav)
            
        # Concatenate frames if the model returned multiple (though typically it processes as one stream unless chunked logic is added)
        # We are just grabbing the codes (index 0 of the tuple)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        
        return codes

    def save_tokens(self, tokens: torch.Tensor, output_path: str):
        """
        Save tokens to a file (PyTorch tensor).
        """
        torch.save(tokens, output_path)
        print(f"Tokens saved to {output_path}")

    def save_tokens_csv(self, tokens: torch.Tensor, output_path: str):
        """
        Save tokens to a human-readable CSV file.
        Format: FrameIndex, Layer1, Layer2, ...
        """
        # Tokens shape: (Batch, Codebooks, Frames)
        # We assume Batch=1 for now
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0) # (Codebooks, Frames)
            
        tokens_np = tokens.cpu().numpy().T # (Frames, Codebooks)
        
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            header = ["Frame"] + [f"Layer_{i+1}" for i in range(tokens_np.shape[1])]
            writer.writerow(header)
            
            # Rows
            for i, row in enumerate(tokens_np):
                writer.writerow([i] + row.tolist())
                
        print(f"Tokens exported to CSV: {output_path}")

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to audio (for verification).
        
        Args:
            tokens: Tensor of shape (Batch, Quantizers, TimeSteps)
            
        Returns:
            torch.Tensor: Reconstructed audio waveform (Batch, Channels, Time)
        """
        # We need to construct the list of (codes, scale) format EnCodec expects.
        # EnCodec 24khz model doesn't use scale (it's None for this model usually, or strictly internal).
        # The encode method returns [(codes, None)].
        
        encoded_frames = [(tokens, None)]
        
        with torch.no_grad():
            audio_values = self.model.decode(encoded_frames)
            
        return audio_values

    def calculate_audio_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """
        Calculate audio quality metrics between original and reconstructed waveforms.
        """
        # Ensure same device and dimensions
        if original.device != reconstructed.device:
            reconstructed = reconstructed.to(original.device)
            
        # 1. Align waveforms (Cross-Correlation)
        # EnCodec may introduce sample shifts/padding.
        # Use center chunk to find offset.
        sr = self.model.sample_rate
        search_window = int(2.0 * sr) # +/- 2 second search
        
        # Convert to mono for alignment if stereo
        tgt_mono = original.mean(dim=0, keepdim=True) if original.shape[0] > 1 else original
        est_mono = reconstructed.mean(dim=0, keepdim=True) if reconstructed.shape[0] > 1 else reconstructed
        
        # Take a representative chunk from the middle of the Original
        mid_idx = original.shape[-1] // 2
        chunk_len = int(0.2 * sr) # Reduced from 5.0 for speed
        start_idx = max(0, mid_idx - chunk_len // 2)
        end_idx = min(original.shape[-1], mid_idx + chunk_len // 2)
        ref_chunk = tgt_mono[..., start_idx:end_idx]
        
        # Search in Reconstructed around the same area +/- search_window
        recon_start = max(0, start_idx - search_window)
        recon_end = min(reconstructed.shape[-1], end_idx + search_window)
        query_chunk = est_mono[..., recon_start:recon_end]
        
        if ref_chunk.abs().sum() > 1e-6:
             # Cross-correlation via conv1d
             # Kernel: ref_chunk (flipped for convolution vs correlation, but alignment is diff)
             # actually correlation(x, y) = conv(x, flip(y)) 
             # We want to find where ref_chunk fits in query_chunk
             import torch.nn.functional as F
             
             # Kernel must be (Out, In, W). Here (1, 1, W)
             kernel = ref_chunk.view(1, 1, -1)
             input_signal = query_chunk.view(1, 1, -1)
             
             # Use conv1d.
             # Note: technically correlation doesn't flip, conv does. But standard cross-corr finding usually involves one.
             # If we want simple match:
             # Match 'ref' inside 'query'.
             out = F.conv1d(input_signal, kernel)
             best_idx = torch.argmax(out)
             
             # Offset calc
             # In query_chunk, the match starts at best_idx.
             # So Reconstructed[recon_start + best_idx] aligns with Original[start_idx]
             offset = (recon_start + best_idx) - start_idx
             
             print(f"Computed Alignment Offset: {offset.item()} samples")
             
             # Apply Shift
             if offset > 0:
                 # Recon is delayed (starts late). Shift Recon left (drop start)
                 reconstructed = reconstructed[..., offset:]
             elif offset < 0:
                 # Recon is early. Pad Recon start or shift Original?
                 # Easier: Shift Original left (drop start) ? No.
                 # Shift Recon right? (Pad start)
                 reconstructed = torch.cat([torch.zeros_like(reconstructed[..., :int(-offset)]), reconstructed], dim=-1)
                 
        # Truncate to minimum length
        min_len = min(original.shape[-1], reconstructed.shape[-1])
        orig_trim = original[..., :min_len]
        recon_trim = reconstructed[..., :min_len]
        
        # 2. Si-SNR (Scale-Invariant Signal-to-Noise Ratio)
        # Definition: 10 * log10( ||s_target||^2 / ||e_noise||^2 )
        # where s_target = <x, s> * s / ||s||^2  (projection of x onto s)
        # ignoring optimal scaling for simplicity as EnCodec usually preserves scale well enough, 
        # or implementing full Si-SNR logic.
        
        # Let's implement standard Si-SNR
        target = orig_trim
        estimate = recon_trim
        
        # Zero mean
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        
        # Dot product
        dot = torch.sum(target * estimate, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
        
        # Projection of estimate onto target
        projection = (dot / target_energy) * target
        
        # Noise component
        noise = estimate - projection
        
        # Energies
        signal_energy = torch.sum(projection ** 2, dim=-1) + 1e-8
        noise_energy = torch.sum(noise ** 2, dim=-1) + 1e-8
        
        si_snr = 10 * torch.log10(signal_energy / noise_energy)
        si_snr_val = si_snr.mean().item()
        
        # 3. Correlation (Pearson)
        vx = target - torch.mean(target, dim=-1, keepdim=True)
        vy = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        correlation = torch.sum(vx * vy, dim=-1) / (torch.sqrt(torch.sum(vx ** 2, dim=-1) * torch.sum(vy ** 2, dim=-1)) + 1e-8)
        corr_val = correlation.mean().item()
        
        return {"Si-SNR": si_snr_val, "Correlation": corr_val}

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Encodec Audio Tokenizer")
    parser.add_argument("audio_path", help="Path to input audio file")
    # Optional output argument, if not provided we auto-generate
    parser.add_argument("--output", help="Optional path to output token file (.pt). If omitted, saves to 'outputs/Filename_tokens.pt'")
    
    # Bandwidth argument
    # EnCodec supports 1.5, 3.0, 6.0, 12.0, 24.0
    parser.add_argument("--bandwidth", type=float, default=24.0, choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                        help="Target bandwidth in kbps. Higher = Better Quality. Default: 24.0")
    
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found try to check they file once : {audio_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Define output directory relative to this script
        script_dir = Path(__file__).parent
        output_dir = script_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filename: SongName_TIMESTAMP_tokens.pt
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use stem (filename without extension)
        output_filename = f"{audio_path.stem}_{timestamp}_tokens.pt"
        output_path = output_dir / output_filename

    tokenizer = AudioTokenizer(target_bandwidth=args.bandwidth)
    
    # Need to load original audio manually here for comparison later if not handled by tokenize
    # But tokenize loads it internally. Let's load it here to have ground truth "original" tensor.
    import soundfile as sf
    wav_np, sr = sf.read(str(audio_path))
    original_wav = torch.from_numpy(wav_np).float()
    if original_wav.dim() == 1: original_wav = original_wav.unsqueeze(0)
    else: original_wav = original_wav.t()
    
    # Resample original to 48kHz for fair comparison (EnCodec operates at 48kHz typically)
    if sr != tokenizer.model.sample_rate:
         original_wav = convert_audio(original_wav.unsqueeze(0), sr, tokenizer.model.sample_rate, tokenizer.model.channels).squeeze(0)

    
    tokens = tokenizer.tokenize(str(audio_path))
    print(f"Generated tokens shape: {tokens.shape}")
    
    # --- Token Statistics ---
    # Shape is (Batch, Codebooks, Frames)
    num_codebooks = tokens.shape[1]
    num_frames = tokens.shape[2]
    
    # Duration in seconds = Total Samples / Sample Rate
    # original_wav was loaded earlier
    duration_sec = original_wav.shape[-1] / tokenizer.model.sample_rate
    
    if duration_sec > 0:
        tokens_per_sec = num_frames / duration_sec
    else:
        tokens_per_sec = 0
        
    total_tokens = num_codebooks * num_frames
    
    print("\n--- Token Statistics ---")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Codebooks (Layer Depth): {num_codebooks}")
    print(f"Frames (Time Steps): {num_frames}")
    print(f"Tokens per Second (Frame Rate): {tokens_per_sec:.2f} Hz")
    print(f"Total Tokens Generated: {total_tokens} ( = {num_frames} Frames * {num_codebooks} Codebooks)")
    print("------------------------\n")
    
    tokenizer.save_tokens(tokens, str(output_path))
    
    # Also save as CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    tokenizer.save_tokens_csv(tokens, str(csv_path))
    
    # Decode back to audio for verification
    print("Decoding tokens back to audio...")
    reconstructed_audio = tokenizer.decode(tokens)
    
    # Remove batch dimension: (1, Channels, Time) -> (Channels, Time)
    reconstructed_audio = reconstructed_audio.squeeze(0)
    
    # Save reconstructed audio
    # Use output_path's stem (which includes timestamp) to match filenames
    reconstructed_filename = f"{output_path.stem.replace('_tokens', '')}_reconstructed.wav"
    reconstructed_path = output_dir / reconstructed_filename
    
    # torchaudio.save might fail if backend issues exist, use soundfile
    import soundfile as sf
    # soundfile expects (Time, Channels), tensor is (Channels, Time)
    audio_np = reconstructed_audio.cpu().numpy().T
    sf.write(reconstructed_path, audio_np, tokenizer.model.sample_rate)
    
    print(f"Original Audio: {audio_path}")
    print(f"Reconstructed Audio: {reconstructed_path}")
    print("Please listen to both files to compare quality.")
    
    # Calculate and print metrics
    print("\n--- Audio Quality Metrics ---")
    
    
    # Original is (Channels, Time) loaded manually, Recon is (Channels, Time)
    metrics = tokenizer.calculate_audio_metrics(original_wav, reconstructed_audio.cpu())
    
    # Qualitative Labels.    Scale-Invariant Signal-to-Noise Ratio.  gm 
    
    si_snr = metrics['Si-SNR']
    if si_snr > 20: snr_label = "Excellent"
    elif si_snr > 10: snr_label = "Good"
    elif si_snr > 3: snr_label = "Fair"
    else: snr_label = "Poor"
    
    corr = metrics['Correlation']
    if corr > 0.95: corr_label = "Excellent"
    elif corr > 0.85: corr_label = "Good"
    elif corr > 0.70: corr_label = "Fair"
    else: corr_label = "Poor"
    
    print(f"Si-SNR: {si_snr:.2f} dB ({snr_label})")
    print(f"Waveform Correlation: {corr:.4f} ({corr_label})")
