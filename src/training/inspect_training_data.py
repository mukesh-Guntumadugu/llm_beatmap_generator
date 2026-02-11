import torch
import glob
import os

def inspect_data():
    search_path = "src/Neural Audio Codecs/outputs/*_dataset.pt"
    files = glob.glob(search_path)
    
    if not files:
        print(f"No dataset files found in {search_path}")
        return

    print(f"Found {len(files)} dataset files.")
    
    for fpath in files:
        print("-" * 50)
        print(f"Inspecting: {os.path.basename(fpath)}")
        try:
            data = torch.load(fpath)
            print(f"Keys: {list(data.keys())}")
            
            tokens = None
            targets = None
            
            if 'tokens' in data:
                tokens = data['tokens']
                targets = data['targets']
            elif 'audio_tokens' in data:
                tokens = data['audio_tokens']
                targets = data['beatmap_labels']
            
            if tokens is not None:
                print(f"Tokens Shape: {tokens.shape}")
                print(f"Tokens Type: {tokens.dtype}")
                # EnCodec tokens are usually (Codebooks, Time)
                if tokens.dim() > 1:
                     print(f"Number of Codebooks: {tokens.shape[0]}")
                     print(f"Sequence Length: {tokens.shape[-1]}")
                
                print(f"Sample Tokens (first 5 time steps across all codebooks):")
                if tokens.dim() == 2:
                    print(tokens[:, :5])
                elif tokens.dim() == 3:
                    print(tokens[0, :, :5])

            if targets is not None:
                print(f"Targets Shape: {targets.shape}")
                print(f"Targets Type: {targets.dtype}")
                unique_targets = torch.unique(targets)
                print(f"Unique Target Values (Classes): {unique_targets.tolist()}")
                print(f"Number of Unique Targets: {len(unique_targets)}")
                
                print(f"Sample Targets (first 20): {targets[:20].tolist()}")
                
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
if __name__ == "__main__":
    inspect_data()
