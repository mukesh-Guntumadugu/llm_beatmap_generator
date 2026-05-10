#!/usr/bin/env python3
"""
analyze_bpm_error.py
====================
Calculates the Mean Absolute Error (MAE) and exact accuracy of the 
LLM's zero-shot BPM predictions compared to the ground truth.
"""

import os
import csv

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(PROJ_DIR, "outputs", "zero_shot_results.csv")

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find {CSV_FILE}")
        return

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
            gt_bpm_str = row["gt_bpm"]
            pred_bpm_str = row["pred_bpm"]
            
            if model not in model_errors:
                model_errors[model] = 0.0
                model_counts[model] = 0
                
            try:
                # Some models might spit out text instead of a pure number
                gt_bpm = float(gt_bpm_str)
                # Clean up any weird LLM text if it failed to just output a number
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
