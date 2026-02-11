import torch
import sys
import os

def explain_file(path):
    print(f"\n--- Analyzing: {os.path.basename(path)} ---")
    try:
        data = torch.load(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if isinstance(data, torch.Tensor):
        print("TYPE: Raw Tensor (Just Numbers)")
        print(f"SHAPE: {data.shape}")
        print("CONTENTS: This likely contains ONLY Audio Tokens (Inference Input).")
        print("          There is NO Beatmap data here.")
        
    elif isinstance(data, dict):
        print(f"TYPE: Dictionary (Named Data)")
        print(f"KEYS: {list(data.keys())}")
        
        if 'tokens' in data and 'targets' in data:
            print("\nVERDICT: This is a TRAINING DATASET.")
            print("1. 'tokens':  Audio Features (Input)")
            print("2. 'targets': Beatmap Labels (The Human Answer)")
            print(f"   - Targets Shape: {data['targets'].shape}")
            print(f"   - Sample Target: {data['targets'][0].tolist()}")
        elif 'audio_tokens' in data and 'beatmap_labels' in data:
             print("\nVERDICT: This is an OLDER TRAINING DATASET.")
             print("It has both Audio and Beatmap data.")
        else:
             print("\nVERDICT: This is just a collection of data, but maybe not a full dataset.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/training/explain_pt_file.py <path_to_file.pt>")
    else:
        explain_file(sys.argv[1])
