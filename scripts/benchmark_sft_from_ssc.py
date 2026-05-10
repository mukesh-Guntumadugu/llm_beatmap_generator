#!/usr/bin/env python3
"""
benchmark_sft_from_ssc.py
=========================
Parses the generated `.ssc` files in the `outputs/` directory to evaluate 
SFT performance for BPM and Onsets.

Usage:
  python scripts/benchmark_sft_from_ssc.py
"""

import os
import sys
import csv
import glob
import re
import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJ_DIR, "outputs")
DATASET_DIR = os.path.join(PROJ_DIR, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements")
OUT_CSV = os.path.join(OUTPUTS_DIR, "sft_eval_results.csv")

def extract_sft_metrics_from_ssc(ssc_path):
    """Parses a generated .ssc file to extract Predicted BPM and Predicted Onsets."""
    try:
        sf = simfile.open(ssc_path)
    except Exception as e:
        return "ERROR", 0

    # 1. Extract BPM
    # Format of BPMS string is "beat=bpm,beat=bpm,..."
    bpms_str = sf.bpms
    if not bpms_str:
        pred_bpm = "UNKNOWN"
    else:
        # Just grab the first BPM value for simplicity
        bpm_parts = bpms_str.split("=")
        if len(bpm_parts) >= 2:
            pred_bpm = bpm_parts[1].split(",")[0]
        else:
            pred_bpm = "UNKNOWN"

    # 2. Extract Onsets (count of all tap/hold/roll/mine heads)
    total_onsets = 0
    # Search across all charts/difficulties
    for chart in sf.charts:
        notes = chart.notes
        # Strip comments and metadata, keep only notes
        clean_notes = re.sub(r'//.*', '', notes)
        clean_notes = clean_notes.replace(' ', '').replace('\\n', '').replace('\\r', '')
        # Notes are mapped 4 chars at a time.
        # We consider '1' (tap), '2' (hold head), '4' (roll head), 'M' (mine) as an onset.
        for char in clean_notes:
            if char in ['1', '2', '4', 'M']:
                total_onsets += 1
                
        # If there are multiple difficulties, we'll just sum them up or take the Challenge one?
        # Actually, for onset prediction benchmarking, usually we take the Challenge or Hard diff.
        # Let's just grab the highest difficulty onset count.
        
    return pred_bpm, total_onsets

def get_ground_truth(song_name):
    gt_onsets = 0
    song_dir = os.path.join(DATASET_DIR, song_name)
    csvs = glob.glob(os.path.join(song_dir, "original_onsets_*.csv"))
    if csvs:
        csv_path = max(csvs, key=os.path.getmtime)
        try:
            with open(csv_path, "r") as f:
                gt_onsets = sum(1 for _ in f) - 1 # skip header
        except: pass
    return gt_onsets

def main():
    print(f"Scanning {OUTPUTS_DIR} for generated .ssc files...")
    ssc_files = glob.glob(os.path.join(OUTPUTS_DIR, "*.ssc"))
    
    if not ssc_files:
        print("No .ssc files found in outputs/ directory. Run SFT inference first!")
        return
        
    print(f"Found {len(ssc_files)} generated charts.")
    
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ssc_file", "model", "song_name", "gt_onsets", "pred_onsets", "pred_bpm"])
        
        for ssc_path in ssc_files:
            filename = os.path.basename(ssc_path)
            # Filenames usually look like: qwen_5diff_Bad_Ketchup.ssc
            # We try to infer model and song name
            parts = filename.replace(".ssc", "").split("_", 2)
            model_name = parts[0].upper()
            if len(parts) == 3:
                song_name = parts[2].replace("_", " ")
            else:
                song_name = "Unknown"
                
            pred_bpm, pred_onsets = extract_sft_metrics_from_ssc(ssc_path)
            gt_onsets = get_ground_truth(song_name)
            
            print(f"Parsed {filename}: Model={model_name}, Song={song_name}, PredOnsets={pred_onsets}, GTOnsets={gt_onsets}, BPM={pred_bpm}")
            writer.writerow([filename, model_name, song_name, gt_onsets, pred_onsets, pred_bpm])

    print(f"\\n✅ SFT Benchmarking complete! Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
