import torch
import sys
import os
import pandas as pd
import numpy as np

def dump_to_csv(pt_path):
    print(f"Loading {pt_path}...")
    try:
        data = torch.load(pt_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    tokens = None
    targets = None

    if 'tokens' in data:
        tokens = data['tokens'].cpu().numpy()
        if 'targets' in data:
            targets = data['targets'].cpu().numpy()
    elif 'audio_tokens' in data:
        tokens = data['audio_tokens'].cpu().numpy()
        if 'beatmap_labels' in data:
            targets = data['beatmap_labels'].cpu().numpy()
    
    if tokens is None:
        print("Could not find 'tokens' or 'audio_tokens' in file.")
        if isinstance(data, torch.Tensor):
            print("File is a raw tensor (likely just tokens).")
            tokens = data.cpu().numpy()
        else:
            print(f"Keys found: {data.keys()}")
            return

    # Squeeze dimensions if needed
    # Check shape: (Layers, Time) vs (Batch, Layers, Time)
    print(f"Raw Tokens Shape: {tokens.shape}")
    
    # We want (Time, Layers)
    if tokens.ndim == 3:
        tokens = tokens[0] # Take first batch
    if tokens.shape[0] == 32: # (Layers, Time) -> (Time, Layers)
         tokens = tokens.T
         
    print(f"Processed Tokens Shape: {tokens.shape}")
    
    # Prepare Dataframe
    df_data = {}
    
    # Add Token Columns
    for i in range(tokens.shape[1]):
        df_data[f'Token_L{i}'] = tokens[:, i]
        
    # Add Target Columns if they exist
    if targets is not None:
        print(f"Raw Targets Shape: {targets.shape}")
        # Targets might be (Time, Lanes) or (Batch, Time, Lanes)
        if targets.ndim == 3:
             targets = targets[0] 
        
        # Ensure length matches
        min_len = min(tokens.shape[0], targets.shape[0])
        
        # Add target columns
        for i in range(targets.shape[1]): # Should be 4 lanes
            df_data[f'Lane_{i}'] = targets[:min_len, i]
            
        print(f"Processed Targets Shape: {targets.shape}")
    else:
        print("NO BEATMAP TARGETS FOUND IN FILE!")

    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Output CSV
    csv_path = pt_path.replace('.pt', '.csv')
    df.to_csv(csv_path, index_label='TimeStep')
    print(f"\nSuccessfully saved dump to: {csv_path}")
    
    # Validation Print
    if targets is not None:
        print("\n--- BEATMAP DATA CHECK ---")
        non_zero_rows = df[(df['Lane_0'] > 0) | (df['Lane_1'] > 0) | (df['Lane_2'] > 0) | (df['Lane_3'] > 0)]
        if not non_zero_rows.empty:
            print(f"Found {len(non_zero_rows)} rows with notes.")
            print("Sample of non-zero rows:")
            print(non_zero_rows[['Lane_0', 'Lane_1', 'Lane_2', 'Lane_3']].head())
        else:
            print("WARNING: All beatmap lanes are ZERO (Empty chart?)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dump_pt_to_csv.py <path_to_file.pt>")
    else:
        dump_to_csv(sys.argv[1])
