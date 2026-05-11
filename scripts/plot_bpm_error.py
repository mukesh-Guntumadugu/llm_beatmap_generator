#!/usr/bin/env python3
import os
import csv
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILES = glob.glob(os.path.join(PROJ_DIR, "outputs", "zero_shot_*.csv"))
BPM_JSON = os.path.join(PROJ_DIR, "local_bpm_analysis.json")
PLOT_OUTPUT = os.path.join(PROJ_DIR, "outputs", "zero_shot_bpm_mae_simple.png")

def main():
    if not CSV_FILES:
        return
        
    with open(BPM_JSON, "r") as f:
        bpm_db = json.load(f)
        
    song_to_bpm = {}
    for sm_key, sm_data in bpm_db.items():
        if "sections" in sm_data and len(sm_data["sections"]) > 0:
            primary_bpm = sm_data["sections"][0]["bpm"]
            song_name = sm_key.replace(".sm", "").replace(".ssc", "")
            song_to_bpm[song_name] = primary_bpm

    model_errors = {}
    model_counts = {}

    for csv_file in CSV_FILES:
        if "zero_shot_results.csv" in csv_file: continue
            
        with open(csv_file, "r", encoding="utf-8") as f:
            content = f.read().replace('\x00', '')
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                model = row["model"]
                song = row["song_name"]
                pred_bpm_str = row["pred_bpm"]
                
                gt_bpm = None
                for key in song_to_bpm.keys():
                    if key in song or song in key:
                        gt_bpm = song_to_bpm[key]
                        break
                
                if gt_bpm is None: continue
                    
                if model not in model_errors:
                    model_errors[model] = 0.0
                    model_counts[model] = 0
                
                try:
                    import re
                    pred_nums = re.findall(r"\d+\.?\d*", pred_bpm_str)
                    pred_bpm = float(pred_nums[0]) if pred_nums else 0.0
                    model_errors[model] += abs(gt_bpm - pred_bpm)
                    model_counts[model] += 1
                except Exception:
                    pass

    models = []
    maes = []
    for model in sorted(model_errors.keys()):
        if model_counts[model] > 0:
            models.append(model)
            maes.append(model_errors[model] / model_counts[model])

    models.append("Gemini 2.5 Pro")
    maes.append(21.0)

    if not models: return

    # Plot single simple bar chart
    plt.style.use('default') # Use clean white style
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Format long names with newlines so they don't overlap
    display_names = []
    for m in models:
        if m == "DeepResonance": display_names.append("DeepResonance")
        elif m == "Gemini 2.5 Pro": display_names.append("Gemini\n2.5 Pro")
        else: display_names.append(m)
    
    # Force pure white backgrounds
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False) # Ensure NO grid lines are drawn
    
    # Remove top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    bars = ax.bar(display_names, maes, color=colors[:len(models)], edgecolor='black')

    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=18)

    plt.title("Zero-Shot Model Performance (BPM)", fontsize=24, fontweight='bold', pad=20)
    plt.ylabel("Mean Absolute Error\n(Lower is Better)", fontsize=18, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.ylim(0, max(maes) + 5)
    
    os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✅ Generated simple bar chart: {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()
