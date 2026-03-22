"""
Evaluate the fine-tuned Qwen2-Audio onset detector against
librosa ground-truth (original_onsets_*.csv).

Usage:
    # Run on a single hold-out song (must NOT be in training set):
    python3 scripts/evaluate_onset_f1.py --song "Girls" --server http://localhost:8000

    # Run on multiple hold-out songs:
    python3 scripts/evaluate_onset_f1.py \\
        --songs "Girls" "Goin' Under" "Goodbye (2012 Mix)" \\
        --server http://localhost:8000

    # Compare against a pre-existing qwen_onsets/*.txt file (no server needed):
    python3 scripts/evaluate_onset_f1.py \\
        --song "Girls" \\
        --pred-file "src/musicForBeatmap/Fraxtil's Arrow Arrangements/Girls/qwen_onsets/Qwen_LoRA_onsets_Girls_XXXX.txt"

The script prints Precision, Recall, F1 (and optionally saves a results JSON).
A "match" is defined as: |predicted_ms - ground_truth_ms| <= TOLERANCE_MS
"""

import os
import sys
import re
import glob
import argparse
import json
import datetime
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE_DIR      = os.path.join(_PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
TOLERANCE_MS  = 50.0   # Match window: ±50 ms (standard in onset detection research)


# ── Ground-truth loader ───────────────────────────────────────────────────────
def load_ground_truth(song_dir: str) -> list[float]:
    """Load the librosa original_onsets_*.csv and return timestamps in ms."""
    csv_files = glob.glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if not csv_files:
        return []
    csv_files.sort()  # Take the most recent one if multiple
    csv_path = csv_files[-1]
    print(f"     [GT] Using ground truth file: {os.path.basename(csv_path)}")
    
    onsets_ms = []
    with open(csv_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                onsets_ms.append(float(parts[1]))
            except ValueError:
                pass
    return sorted(onsets_ms)


# ── Prediction loader ─────────────────────────────────────────────────────────
def load_predictions_from_file(pred_file: str) -> list[float]:
    """Load raw millisecond timestamps from a qwen_onsets/*.txt file."""
    onsets_ms = []
    with open(pred_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                onsets_ms.append(float(line))
            except ValueError:
                pass
    return sorted(onsets_ms)


def load_latest_prediction(song_dir: str) -> list[float]:
    """Load the latest Qwen_LoRA_onsets_*.txt from qwen_onsets/ subfolder."""
    pred_dir = os.path.join(song_dir, "qwen_onsets")
    txt_files = glob.glob(os.path.join(pred_dir, "Qwen_LoRA_onsets_*.txt"))
    if not txt_files:
        return []
    txt_files.sort()  # Latest by name/timestamp
    latest_file = txt_files[-1]
    print(f"     [Qwen] Using prediction file: {os.path.basename(latest_file)}")
    return load_predictions_from_file(latest_file)


# ── F1 Scorer ─────────────────────────────────────────────────────────────────
def score_onsets(predictions_ms: list[float],
                 ground_truth_ms: list[float],
                 tolerance_ms: float = TOLERANCE_MS) -> dict:
    """
    Calculates Precision, Recall, F1 using a greedy match:
    each GT onset can only be matched to ONE predicted onset.
    """
    if not predictions_ms or not ground_truth_ms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": len(predictions_ms), "fn": len(ground_truth_ms)}

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
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp":  tp, "fp": fp, "fn": fn,
        "n_predicted":   len(pred_arr),
        "n_ground_truth": len(gt_arr),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen onset detection F1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--song",  help="Single song name to evaluate")
    group.add_argument("--songs", nargs="+", help="Multiple song names to evaluate")
    group.add_argument("--all",   action="store_true", help="Evaluate ALL songs with predictions")

    parser.add_argument("--pred-file", default=None,
                        help="Path to a specific prediction .txt file (for single --song only)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_MS,
                        help=f"Match tolerance in ms (default: {TOLERANCE_MS})")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to save full results as JSON")
    args = parser.parse_args()

    # Build song list
    if args.all:
        song_names = sorted([
            os.path.basename(d) for d in glob.glob(os.path.join(BASE_DIR, "*"))
            if os.path.isdir(d)
        ])
    elif args.songs:
        song_names = args.songs
    else:
        song_names = [args.song]

    all_results = {}
    total_tp = total_fp = total_fn = 0

    print(f"\n{'='*60}")
    print(f"  Qwen LoRA Onset Detection — F1 Evaluation")
    print(f"  Tolerance: ±{args.tolerance} ms")
    print(f"{'='*60}\n")

    for song_name in song_names:
        print(f"  🎵 {song_name}")
        song_dir = os.path.join(BASE_DIR, song_name)
        if not os.path.isdir(song_dir):
            print(f"     ⚠️  Song directory not found: {song_name}")
            continue

        gt_ms = load_ground_truth(song_dir)
        if not gt_ms:
            print(f"⚠️  No ground truth CSV for: {song_name} — skipping")
            continue

        # Load predictions
        if args.pred_file and len(song_names) == 1:
            print(f"     [Qwen] Using specific prediction file: {os.path.basename(args.pred_file)}")
            pred_ms = load_predictions_from_file(args.pred_file)
        else:
            pred_ms = load_latest_prediction(song_dir)

        if not pred_ms:
            print(f"⚠️  No prediction file found for: {song_name} — skipping")
            continue

        scores = score_onsets(pred_ms, gt_ms, args.tolerance)
        all_results[song_name] = scores

        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

        f1_bar = "█" * int(scores["f1"] * 20)
        print(f"     GT: {scores['n_ground_truth']} onsets | Predicted: {scores['n_predicted']} onsets")
        print(f"     TP: {scores['tp']}  FP: {scores['fp']}  FN: {scores['fn']}")
        print(f"     Precision: {scores['precision']:.1%}  |  Recall: {scores['recall']:.1%}  |  F1: {scores['f1']:.1%}  [{f1_bar:<20}]")
        print()

    # Overall micro-average
    if total_tp + total_fp > 0 and total_tp + total_fn > 0:
        micro_p = total_tp / (total_tp + total_fp)
        micro_r = total_tp / (total_tp + total_fn)
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        print(f"{'='*60}")
        print(f"  OVERALL MICRO-AVERAGE ({len(all_results)} songs)")
        print(f"  Precision: {micro_p:.1%}  |  Recall: {micro_r:.1%}  |  F1: {micro_f1:.1%}")
        print(f"{'='*60}\n")

        if micro_f1 >= 0.85:
            print("  ✅ EXCELLENT training! F1 ≥ 85%")
        elif micro_f1 >= 0.70:
            print("  👍 GOOD training. F1 ≥ 70%")
        elif micro_f1 >= 0.50:
            print("  ⚠️  MODERATE training. Consider more data or epochs.")
        else:
            print("  ❌ POOR training. Model needs more data or longer training.")

    # Save JSON
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({
                "timestamp": datetime.datetime.now().isoformat(),
                "tolerance_ms": args.tolerance,
                "songs": all_results,
                "overall": {
                    "precision": round(micro_p, 4),
                    "recall":    round(micro_r, 4),
                    "f1":        round(micro_f1, 4),
                }
            }, f, indent=2)
        print(f"\n  📄 Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
