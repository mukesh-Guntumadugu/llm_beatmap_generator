import torch
import argparse
import sys

def view_tokens(token_path):
    try:
        # Load the tokens
        tokens = torch.load(token_path, weights_only=False) # weights_only=False to support older formats if needed, though usually True is safer. 
        # For simple tensors True is fine, but sometimes these are lists/tuples. 
        # EnCodec output was a tensor, so let's try default or just load.
        
        print(f"--- Token File: {token_path} ---")
        print(f"Type: {type(tokens)}")
        
        if isinstance(tokens, torch.Tensor):
            print(f"Shape: {tokens.shape}")
            print(f"Dtype: {tokens.dtype}")
            print("\nContent:")
            torch.set_printoptions(threshold=sys.maxsize, linewidth=200)
            print(tokens)
        elif isinstance(tokens, dict):
            print("Type: Dictionary (Dataset)")
            for k, v in tokens.items():
                print(f"\n--- Key: {k} ---")
                if isinstance(v, torch.Tensor):
                    print(f"Shape: {v.shape}")
                    print(f"Dtype: {v.dtype}")
                    # Prepare print options
                    torch.set_printoptions(threshold=1000, linewidth=200)
                    
                    if k == 'beatmap_labels' and v.shape[1] == 4:
                        print("Content (First 500 rows where sum > 0):")
                        # Show only active rows for readability
                        active_mask = v.sum(dim=1) > 0
                        active_indices = torch.where(active_mask)[0]
                        if len(active_indices) > 0:
                            limit = min(500, len(active_indices))
                            for idx in active_indices[:limit]:
                                print(f"TimeStep {idx}: {v[idx].tolist()}")
                        else:
                            print("No active beats found.")
                    else:
                        print(v)
                else:
                    print(v)
        else:
            print(tokens)
            
    except Exception as e:
        print(f"Error loading {token_path}: {e}")

if __name__ == "__main__":
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="View content of .pt token files")
    parser.add_argument("token_path", help="Path to the .pt file")
    args = parser.parse_args()
    
    token_path = Path(args.token_path)
    
    # Run the view logic
    # We will hijack stdout to capture output or rewrite view_tokens to return a string
    # For simplicity, let's redirect stdout to both console and file
    
    # Determine output text path
    # If the file is in 'outputs/Song.pt', we want 'outputs/Song.txt'
    output_txt_path = token_path.with_suffix(".txt")
    
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self
        def __del__(self):
            sys.stdout = self.stdout
            self.file.close()
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()

    print(f"Viewing tokens from: {token_path}")
    print(f"Saving readable text to: {output_txt_path}")
    
    # Start Teeing output
    tee = Tee(output_txt_path, "w")
    try:
        view_tokens(str(token_path))
    finally:
        del tee
