#!/usr/bin/env python3
import os
import glob
import json
import csv
import argparse

def evaluate_onsets(true_onsets, pred_onsets, tolerance=50):
    """
    Evaluates predicted onsets against true onsets.
    A predicted onset is a 'Hit' (True Positive) if it is within 'tolerance' ms of a true onset.
    Matches are 1-to-1 max (greedy matching).
    """
    true_unmatched = sorted(true_onsets.copy())
    pred_unmatched = sorted(pred_onsets.copy())
    
    hits = 0
    # Match greedily
    for p in pred_unmatched[:]:
        # Find closest true onset
        if not true_unmatched:
            break
        
        closest_true = min(true_unmatched, key=lambda t: abs(t - p))
        if abs(closest_true - p) <= tolerance:
            hits += 1
            true_unmatched.remove(closest_true)
            pred_unmatched.remove(p)

    false_positives = len(pred_unmatched)
    false_negatives = len(true_unmatched)
    
    precision = hits / len(pred_onsets) if pred_onsets else 0.0
    recall = hits / len(true_onsets) if true_onsets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "hits": hits,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def load_ground_truth(gt_path):
    """Load ground truth from either a JSON (librosa) or CSV (original beatmap)."""
    ext = os.path.splitext(gt_path)[-1].lower()
    if ext == ".json":
        with open(gt_path, 'r') as f:
            data = json.load(f)
        onsets = [int(round(x)) for x in data.get("onsets_ms", [])]
        source = "Librosa JSON"
    elif ext == ".csv":
        onsets = []
        with open(gt_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    onsets.append(int(round(float(row["onset_ms"]))))
                except (KeyError, ValueError):
                    pass
        source = "Original Beatmap CSV"
    else:
        raise ValueError(f"Unsupported ground truth format: {ext}")
    return onsets, source

def main(gt_path, csv_dir, tolerance):
    print("=" * 75)
    print(f"GROUND TRUTH: {os.path.basename(gt_path)}")
    print("=" * 75)
    true_onsets, source = load_ground_truth(gt_path)
    print(f"Total True Onsets ({source}): {len(true_onsets)}")
    
    models = ["DeepResonance", "Qwen", "Flamingo", "MuMu", "Qwen_LoRA"]
    
    for model in models:
        # Find the most recently generated output file for this model
        if model == "Qwen_LoRA":
            # Reads from the dedicated qwen_onsets subfolder (plain text, one ms per line)
            pattern = os.path.join(csv_dir, "qwen_onsets", "Qwen_LoRA_onsets_Bad_Ketchup_*.txt")
        else:
            pattern = os.path.join(csv_dir, f"{model}_TEST_Bad_Ketchup_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"\n[{model.upper()}]")
            print("  → Error: No matching output file found.")
            continue
            
        latest_file = max(files, key=os.path.getmtime)
        pred_onsets = []
        
        with open(latest_file, 'r') as f:
            if latest_file.endswith(".txt"):
                # Plain text: one onset_ms per line, no header
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pred_onsets.append(int(round(float(line))))
                        except ValueError:
                            pass
            else:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0].strip():
                        try:
                            pred_onsets.append(int(float(row[0])))
                        except ValueError:
                            pass
                        
        # Evaluate
        metrics = evaluate_onsets(true_onsets, pred_onsets, tolerance)
        p = metrics['precision'] * 100
        r = metrics['recall'] * 100
        f1 = metrics['f1_score'] * 100
        
        print("-" * 75)
        print(f"MODEL: {model.upper()}")
        print(f"EVAL FILE: {os.path.basename(latest_file)}")
        print(f"  → Accurate Predictions (Hits): {metrics['hits']}")
        print(f"  → Over-predictions (False Positives): {metrics['false_positives']}")
        print(f"  → Missed Beats (False Negatives): {metrics['false_negatives']}")
        print(f"  → Precision: {p:.2f}%")
        print(f"  → Recall:    {r:.2f}%")
        print(f"  → F1-Score:  {f1:.2f}%")
        
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/original_onsets_Bad_Ketchup_15032026143021.csv",
        help="Ground truth: path to original beatmap CSV OR librosa JSON"
    )
    parser.add_argument("--csv_dir", default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup")
    parser.add_argument("--tolerance", type=int, default=50, help="Tolerance in ms")
    args = parser.parse_args()
    
    main(args.gt, args.csv_dir, args.tolerance)
