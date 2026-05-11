#!/usr/bin/env python3
"""
evaluate_all_onsets_f1.py
=========================
Calculates Precision, Recall, and F1 scores for onset predictions across
all four models (Qwen, MuMu, DeepResonance, Flamingo) against Librosa 
ground-truth. It outputs the final metrics matrix and a CSV summary.
"""

import os
import sys
import glob
import csv
import argparse
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR = os.path.join(_PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUT_CSV = os.path.join(_PROJECT_ROOT, "outputs", "onset_f1_results.csv")
TOLERANCE_MS = 50.0

MODELS = {
    "Qwen":          "qwen_onsets",
    "MuMu":          "mumu_onsets",
    "DeepResonance": "deepresonance_onsets",
    "Flamingo":      "flamingo_onsets"
}

def load_ground_truth(song_dir: str) -> list[float]:
    """Load the librosa original_onsets_*.csv and return timestamps in ms."""
    csv_files = glob.glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if not csv_files:
        return []
    csv_files.sort()
    csv_path = csv_files[-1]
    
    onsets_ms = []
    with open(csv_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                onsets_ms.append(float(parts[1]))
            except ValueError:
                pass
    return sorted(onsets_ms)

def load_predictions(song_dir: str, subfolder: str) -> list[float]:
    """Load the latest raw millisecond predictions (.txt or .csv) for a specific model."""
    pred_dir = os.path.join(song_dir, subfolder)
    pred_files = glob.glob(os.path.join(pred_dir, "*.txt")) + glob.glob(os.path.join(pred_dir, "*.csv"))
    if not pred_files:
        return []
    pred_files.sort(key=os.path.getmtime)
    latest_file = pred_files[-1]
    
    onsets_ms = []
    with open(latest_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("onset_ms"): continue
            try:
                onsets_ms.append(float(line))
            except ValueError:
                pass
    return sorted(onsets_ms)

def score_onsets(predictions_ms: list[float], ground_truth_ms: list[float], tolerance_ms: float = TOLERANCE_MS) -> dict:
    if not predictions_ms or not ground_truth_ms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predictions_ms), "fn": len(ground_truth_ms)}

    pred_arr = np.array(predictions_ms)
    gt_arr   = np.array(ground_truth_ms)

    matched_gt = set()
    tp = 0

    for p in pred_arr:
        diffs = np.abs(gt_arr - p)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance_ms and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_arr) - tp
    fn = len(gt_arr)   - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn
    }

def main():
    print(f"Starting Universal Onset F1 Evaluation (Tolerance: ±{TOLERANCE_MS}ms)")
    
    song_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])
    
    # Store aggregated scores per model: tp, fp, fn
    model_aggregates = {m: {"tp": 0, "fp": 0, "fn": 0} for m in MODELS}
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["song_name", "model", "precision", "recall", "f1", "tp", "fp", "fn"])
        
        for song_dir in song_dirs:
            song_name = os.path.basename(song_dir)
            gt_ms = load_ground_truth(song_dir)
            
            if not gt_ms:
                continue
                
            for model_name, subfolder in MODELS.items():
                pred_ms = load_predictions(song_dir, subfolder)
                
                if not pred_ms:
                    # Model failed or hasn't run for this song yet
                    writer.writerow([song_name, model_name, 0.0, 0.0, 0.0, 0, 0, len(gt_ms)])
                    model_aggregates[model_name]["fn"] += len(gt_ms)
                    continue
                    
                scores = score_onsets(pred_ms, gt_ms, TOLERANCE_MS)
                writer.writerow([
                    song_name, model_name, 
                    round(scores["precision"], 4), 
                    round(scores["recall"], 4), 
                    round(scores["f1"], 4),
                    scores["tp"], scores["fp"], scores["fn"]
                ])
                
                model_aggregates[model_name]["tp"] += scores["tp"]
                model_aggregates[model_name]["fp"] += scores["fp"]
                model_aggregates[model_name]["fn"] += scores["fn"]

    # Print final matrix
    print("\n" + "="*80)
    print(f"{'MODEL':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1 SCORE':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 80)
    
    for model_name in MODELS:
        agg = model_aggregates[model_name]
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        
        if tp + fp == 0 and tp + fn == 0:
            print(f"{model_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {tp:<6} | {fp:<6} | {fn:<6}")
            continue
            
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        
        print(f"{model_name:<15} | {p:.2%}     | {r:.2%}     | {f1:.2%}     | {tp:<6} | {fp:<6} | {fn:<6}")
        
    print("="*80)
    print(f"\n✅ Detailed results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
