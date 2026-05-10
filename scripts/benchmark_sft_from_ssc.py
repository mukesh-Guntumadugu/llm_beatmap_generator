#!/usr/bin/env python3
"""
benchmark_sft_from_ssc.py
=========================
Parses the generated `.ssc` files in the `outputs/` directory to evaluate 
SFT performance for BPM and Onsets.

Calculates exact Precision, Recall, and F1 by converting StepMania
beats back into milliseconds via the TimingEngine.
"""

import os
import glob
import csv
import numpy as np

import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine
from simfile.notes import NoteData, NoteType

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJ_DIR, "outputs")
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUT_CSV = os.path.join(OUTPUTS_DIR, "sft_eval_results.csv")
TOLERANCE_MS = 50.0

def load_ground_truth(song_name):
    """Load the librosa original_onsets_*.csv and return timestamps in ms."""
    song_dir = os.path.join(DATASET_DIR, song_name)
    csv_files = glob.glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if not csv_files:
        return []
    csv_path = max(csv_files, key=os.path.getmtime)
    
    onsets_ms = []
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    onsets_ms.append(float(parts[1]))
                except ValueError:
                    pass
    except Exception as e:
        pass
    return sorted(onsets_ms)

def score_onsets(predictions_ms: list[float],
                 ground_truth_ms: list[float],
                 tolerance_ms: float = TOLERANCE_MS) -> dict:
    """Calculates Precision, Recall, F1 using a greedy match."""
    if not predictions_ms or not ground_truth_ms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": len(predictions_ms), "fn": len(ground_truth_ms),
                "n_pred": len(predictions_ms), "n_gt": len(ground_truth_ms)}

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
        "n_pred": len(pred_arr),
        "n_gt":   len(gt_arr),
    }

def extract_sft_metrics_from_ssc(ssc_path):
    """Parses .ssc to extract Predicted BPM and Predicted Onset milliseconds."""
    try:
        sf = simfile.open(ssc_path)
    except Exception as e:
        return "ERROR", []

    # 1. Extract BPM
    bpms_str = sf.bpms
    if not bpms_str:
        pred_bpm = "UNKNOWN"
    else:
        bpm_parts = bpms_str.split("=")
        if len(bpm_parts) >= 2:
            pred_bpm = bpm_parts[1].split(",")[0]
        else:
            pred_bpm = "UNKNOWN"

    # 2. Extract Onsets (in ms)
    pred_onsets_ms = []
    
    # Grab the hardest difficulty chart, or the first available
    # For a fair comparison, if we generated multiple charts, we evaluate the highest note count one
    best_chart = None
    max_notes = -1
    
    for chart in sf.charts:
        if chart.stepstype != 'dance-single':
            continue
        try:
            nd = NoteData(chart)
            count = sum(1 for n in nd if n.note_type in (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.MINE))
            if count > max_notes:
                max_notes = count
                best_chart = chart
        except:
            pass

    if best_chart:
        try:
            # We must use the TimingData from the base simfile (or chart if it has split timing)
            # Standard generated .ssc puts BPMS at the top level
            timing_data = TimingData(sf, best_chart)
            engine = TimingEngine(timing_data)
            nd = NoteData(best_chart)
            
            # Since multiple notes can occur on the exact same beat (jumps), we use a set
            unique_onset_beats = set()
            for note in nd:
                if note.note_type in (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.MINE):
                    unique_onset_beats.add(note.beat)
            
            for beat in unique_onset_beats:
                # time_at returns seconds
                seconds = engine.time_at(beat)
                pred_onsets_ms.append(seconds * 1000.0)
                
            pred_onsets_ms.sort()
        except Exception as e:
            print(f"Error parsing timing data for {ssc_path}: {e}")

    return pred_bpm, pred_onsets_ms

def main():
    print(f"Scanning {OUTPUTS_DIR} for generated .ssc files...")
    ssc_files = glob.glob(os.path.join(OUTPUTS_DIR, "*.ssc"))
    
    if not ssc_files:
        print("No .ssc files found in outputs/ directory. Run SFT inference first!")
        return
        
    print(f"Found {len(ssc_files)} generated charts.")
    
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ssc_file", "model", "song_name", 
            "gt_onsets", "pred_onsets", "pred_bpm",
            "precision", "recall", "f1", "tp", "fp", "fn"
        ])
        
        for ssc_path in ssc_files:
            filename = os.path.basename(ssc_path)
            parts = filename.replace(".ssc", "").split("_", 2)
            model_name = parts[0].upper()
            if len(parts) >= 3:
                # Example: qwen_5diff_Bad_Ketchup
                song_name = parts[2].replace("_", " ")
            else:
                song_name = "Unknown"
                
            pred_bpm, pred_onsets_ms = extract_sft_metrics_from_ssc(ssc_path)
            gt_onsets_ms = load_ground_truth(song_name)
            
            if not gt_onsets_ms:
                # Just skip scoring if we can't find the song dir
                writer.writerow([filename, model_name, song_name, 0, len(pred_onsets_ms), pred_bpm, 0, 0, 0, 0, 0, 0])
                continue
                
            scores = score_onsets(pred_onsets_ms, gt_onsets_ms)
            
            print(f"Parsed {filename}: Model={model_name}, Song={song_name}")
            print(f"  -> GT: {scores['n_gt']} | Pred: {scores['n_pred']} | BPM: {pred_bpm}")
            print(f"  -> Precision: {scores['precision']:.3f} | Recall: {scores['recall']:.3f} | F1: {scores['f1']:.3f}")
            print(f"  -> TP: {scores['tp']} | FP: {scores['fp']} | FN: {scores['fn']}\\n")
            
            writer.writerow([
                filename, model_name, song_name, 
                scores["n_gt"], scores["n_pred"], pred_bpm,
                scores["precision"], scores["recall"], scores["f1"],
                scores["tp"], scores["fp"], scores["fn"]
            ])

    print(f"✅ SFT F1 Benchmarking complete! Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
