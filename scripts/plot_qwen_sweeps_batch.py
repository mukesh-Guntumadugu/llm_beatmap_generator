#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

# Append root to use score logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_onset_detection import load_onsets_csv, score_onsets  # type: ignore

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

def extract_bpm_from_song(song_dir):
    """Scan SSC/SM file for '#BPMS:0.000=XXX' and return BPM float."""
    simfiles = glob.glob(os.path.join(song_dir, "*.ssc")) + glob.glob(os.path.join(song_dir, "*.sm"))
    if not simfiles:
        return None
    
    simfiles.sort(key=lambda x: x.endswith('.ssc'), reverse=True)
    
    with open(simfiles[0], 'r', encoding='utf-8') as f:
        content = f.read()
        
    m = re.search(r"#BPMS:.*?=([\d\.]+)", content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None

def find_latest_file(directory: str, pattern: str):
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

def process_all_songs():
    if not os.path.exists(BASE_DIR):
        print(f"Error: {BASE_DIR} not found.")
        return {}, [], []
        
    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_") and not d.startswith(".")
    ])
    
    chunks = [20, 15, 10, 5, 2]
    static_tolerances = [50, 100, 150, 200, 250, 300]
    
    song_data = {} 
    
    for song in song_dirs:
        sdir = os.path.join(BASE_DIR, song)
        
        bpm = extract_bpm_from_song(sdir)
        if not bpm or bpm <= 0:
            bpm = 120.0
            
        one_beat_ms = (60.0 / bpm) * 1000.0
        tolerances = static_tolerances + [one_beat_ms]
        
        ref_path = find_latest_file(sdir, "original_onsets_*.csv")
        if not ref_path:
            continue
            
        ref_onsets = load_onsets_csv(ref_path)
        if len(ref_onsets) == 0:
            continue
            
        song_data[song] = {
            'bpm': bpm,
            'one_beat_ms': one_beat_ms,
            'metrics': { c: {} for c in chunks }
        }
        
        for ch in chunks:
            patterns = [
                f"*_Qwen_{ch}s_*.csv",
                f"*_Qwen_{ch}sec_*.csv",
                f"qwen_onsets/*_Qwen_{ch}s_*.csv",
                f"qwen_onsets/*_Qwen_{ch}sec_*.csv",
                f"qwen_outputs/*_{ch}s_*.csv",
                f"*Qwen*{ch}*sec*.csv",
                f"*Qwen*{ch}s*.csv",
            ]
            
            pred_path = None
            for pt in patterns:
                found = find_latest_file(sdir, pt)
                if found:
                    pred_path = found
                    break
                    
            if not pred_path:
                continue
                
            pred_onsets = load_onsets_csv(pred_path)
            for tol in tolerances:
                metrics = score_onsets(ref_onsets, pred_onsets, tol)
                song_data[song]['metrics'][ch][tol] = metrics['recall']
                
    return song_data, chunks, static_tolerances

def plot_combined_average(song_data, chunks, static_tolerances, out_file):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan', 'tab:pink']
    
    has_any_data = False
    
    for i, tol in enumerate(static_tolerances):
        y_vals = []
        for ch in chunks:
            recalls = []
            for song, data in song_data.items():
                val = data['metrics'][ch].get(tol, None)
                if val is not None:
                    recalls.append(val)
            if len(recalls) > 0:
                y_vals.append((ch, sum(recalls)/float(len(recalls))))
                
        if len(y_vals) > 0:
            has_any_data = True
            x_idx = [chunks.index(v[0]) for v in y_vals]
            y_pts = [v[1] for v in y_vals]
            ax.plot(x_idx, y_pts, color=colors[i % len(colors)], marker='o', markersize=8, linewidth=2, label=f"{tol}ms")
            
            # Label the points
            for j, val in enumerate(y_pts):
                ax.text(x_idx[j], val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color=colors[i % len(colors)])

    # Dynamic 1-Beat average
    y_vals_dyn = []
    for ch in chunks:
        recalls = []
        for song, data in song_data.items():
            dyn_tol = data['one_beat_ms']
            val = data['metrics'][ch].get(dyn_tol, None)
            if val is not None:
                recalls.append(val)
        if len(recalls) > 0:
            y_vals_dyn.append((ch, sum(recalls)/float(len(recalls))))

    if len(y_vals_dyn) > 0:
        has_any_data = True
        x_idx = [chunks.index(v[0]) for v in y_vals_dyn]
        y_pts = [v[1] for v in y_vals_dyn]
        ax.plot(x_idx, y_pts, color='black', linestyle='--', marker='*', markersize=12, linewidth=2, 
                 label=f"1 Beat (~(60/BPM)*1000 ms mean)")
        for j, val in enumerate(y_pts):
            ax.text(x_idx[j], val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='black')

    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels([f"{ch}s" for ch in chunks])
    ax.set_ylim(-2, 105)
    
    ax.set_xlabel("Audio Chunk Grid Size", fontweight='bold')
    ax.set_ylabel("Average Hit Recall (%)", fontweight='bold')
    ax.set_title("Dataset Average: Qwen2-Audio Onset Tolerance Spread", fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    if has_any_data:
        ax.legend(loc='lower right', shadow=True)
    else:
        ax.text(0.5, 0.5, "No Data / Missing Sweeps", ha='center', va='center', color='gray', fontsize=20, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"\\n📊 Master averaged chart integrated to: {out_file}")

if __name__ == "__main__":
    song_data, chunks, static_tols = process_all_songs()
    if not song_data:
        print("❌ No valid song sweep matrices could be loaded. Aborting.")
        sys.exit(1)
        
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_reports", "Batch_Qwen_Summary_Plot.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_combined_average(song_data, chunks, static_tols, out_path)
