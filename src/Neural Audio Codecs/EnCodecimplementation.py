import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import typing as tp
import os

class AudioTokenizer:
    """
    A class to tokenize audio using Meta's EnCodec model.
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
        Save tokens to a file.
        """
        torch.save(tokens, output_path)
        print(f"Tokens saved to {output_path}")

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

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Encodec Audio Tokenizer")
    parser.add_argument("audio_path", help="Path to input audio file")
    # Optional output argument, if not provided we auto-generate
    parser.add_argument("--output", help="Optional path to output token file (.pt). If omitted, saves to 'outputs/Filename_tokens.pt'")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Define output directory relative to this script
        script_dir = Path(__file__).parent
        output_dir = script_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filename: SongName_tokens.pt
        # Use stem (filename without extension)
        output_filename = f"{audio_path.stem}_tokens.pt"
        output_path = output_dir / output_filename

    tokenizer = AudioTokenizer()
    tokens = tokenizer.tokenize(str(audio_path))
    print(f"Generated tokens shape: {tokens.shape}")
    
    tokenizer.save_tokens(tokens, str(output_path))
