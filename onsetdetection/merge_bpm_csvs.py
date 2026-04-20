"""
merge_bpm_csvs.py
=================
Merges the 5 distinct model/baseline BPM Evaluation CSVs into a single 
unified matrix file.
"""

import os
import pandas as pd

def main():
    base_dir = "onsetdetection"
    models = ["LIBROSA", "QWEN", "MUMU", "DEEPRESONANCE", "FLAMINGO"]
    
    dfs = []
    
    for m in models:
        path = os.path.join(base_dir, f"BPM_Estimates_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Standardize song names just in case
            df['Song_Name'] = df['Song_Name'].str.strip()
            # Set the song name as the index conceptually, but we can do a Pandas merge
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
            
    if not dfs:
        print("No CSVs found to merge!")
        return
        
    master_df = dfs[0]
    for i in range(1, len(dfs)):
        master_df = pd.merge(master_df, dfs[i], on="Song_Name", how="outer")
        
    out_path = os.path.join(base_dir, "Unified_BPM_Benchmark_Results.csv")
    master_df.to_csv(out_path, index=False)
    print(f"✅ Successfully unified {len(dfs)} model logs into:")
    print(f"   -> {out_path}")

if __name__ == "__main__":
    main()
