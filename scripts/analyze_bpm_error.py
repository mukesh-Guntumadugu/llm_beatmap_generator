#!/usr/bin/env python3
"""
analyze_bpm_error.py
====================
Calculates the Mean Absolute Error (MAE) and exact accuracy of the 
LLM's zero-shot BPM predictions compared to the ground truth.
"""

import os
import csv
import json

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(PROJ_DIR, "outputs", "zero_shot_results.csv")
BPM_JSON = os.path.join(PROJ_DIR, "local_bpm_analysis.json")

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find {CSV_FILE}")
        return
        
    if not os.path.exists(BPM_JSON):
        print(f"Error: Could not find {BPM_JSON}")
        return
        
    with open(BPM_JSON, "r") as f:
        bpm_db = json.load(f)
        
    # Map song dir name to its BPM
    song_to_bpm = {}
    for sm_key, sm_data in bpm_db.items():
        if "sections" in sm_data and len(sm_data["sections"]) > 0:
            primary_bpm = sm_data["sections"][0]["bpm"]
            song_name = sm_key.replace(".sm", "").replace(".ssc", "")
            song_to_bpm[song_name] = primary_bpm

    model_errors = {}
    model_counts = {}

    print("========================================")
    print("      ZERO-SHOT BPM ACCURACY REPORT     ")
    print("========================================")
    
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            song = row["song_name"]
            pred_bpm_str = row["pred_bpm"]
            
            # Find the GT BPM
            gt_bpm = None
            for key in song_to_bpm.keys():
                if key in song or song in key:
                    gt_bpm = song_to_bpm[key]
                    break
            
            if gt_bpm is None:
                continue # Skip if we can't find GT
                
            if model not in model_errors:
                model_errors[model] = 0.0
                model_counts[model] = 0
                
            try:
                import re
                pred_nums = re.findall(r"\\d+\\.?\\d*", pred_bpm_str)
                if pred_nums:
                    pred_bpm = float(pred_nums[0])
                else:
                    pred_bpm = 0.0 # Failed prediction
                    
                error = abs(gt_bpm - pred_bpm)
                model_errors[model] += error
                model_counts[model] += 1
                
                print(f"{model:<15} | {song:<30} | GT: {gt_bpm:>6.1f} | Pred: {pred_bpm:>6.1f} | Error: {error:>5.1f}")
            except Exception as e:
                pass

    print("\\n========================================")
    print("      MEAN ABSOLUTE ERROR (BPM)         ")
    print("========================================")
    
    for model in model_errors.keys():
        if model_counts[model] > 0:
            mae = model_errors[model] / model_counts[model]
            print(f"{model:<15} : off by {mae:.1f} Beats Per Minute on average")

if __name__ == "__main__":
    main()
