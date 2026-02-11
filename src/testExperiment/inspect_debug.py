import torch
import sys
from pathlib import Path

def inspect_data():
    dataset_path = "src/testExperiment/Springtime_dataset.pt"
    generated_path = "src/testExperiment/Springtime_generated.txt"
    
    print(f"--- Inspecting Dataset: {dataset_path} ---")
    try:
        data = torch.load(dataset_path)
        targets = data['targets'] # (T, 4)
        
        print(f"Total Frames: {targets.shape[0]}")
        non_zero_indices = torch.nonzero(targets.sum(dim=1)).squeeze()
        print(f"Number of non-zero frames: {len(non_zero_indices)}")
        
        print("\nFirst 10 non-zero frames in DATASET:")
        for idx in non_zero_indices[:10]:
            print(f"Frame {idx.item()}: {targets[idx].tolist()}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

    print(f"\n--- Inspecting Generated File: {generated_path} ---")
    try:
        with open(generated_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if ',' not in l and ';' not in l]
            
        print(f"Total Lines (Frames): {len(lines)}")
        
        non_zero_lines = []
        for i, line in enumerate(lines):
            if line != "0000":
                non_zero_lines.append((i, line))
                
        print(f"Number of non-zero frames: {len(non_zero_lines)}")
        
        print("\nFirst 10 non-zero frames in GENERATED output:")
        for idx, line in non_zero_lines[:10]:
            print(f"Frame {idx}: {line}")
            
    except Exception as e:
        print(f"Error reading generated file: {e}")

if __name__ == "__main__":
    inspect_data()
