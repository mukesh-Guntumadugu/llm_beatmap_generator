import os
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions from the existing script
from score_onset_detection import BASE_DIR, load_onsets_csv, find_latest_file, score_onsets, DEFAULT_TOLERANCE_MS

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: BASE_DIR not found at {BASE_DIR}")
        return

    song_names = []
    gemini_tp = []
    gemini_fp = []
    qwen_tp = []
    qwen_fp = []
    original_counts = []

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and not d.startswith("_") and not d.startswith(".")
    ])
    
    print(f"Processing {len(song_dirs)} directories in {BASE_DIR}...")
    
    # Process each song
    for song in song_dirs:
        song_dir = os.path.join(BASE_DIR, song)
        
        # Original (Ground truth)
        orig_file = find_latest_file(song_dir, "original_onsets_*.csv")
        if not orig_file:
            continue
            
        orig_onsets = load_onsets_csv(orig_file)
        n_orig = len(orig_onsets)
        
        # Gemini
        gemini_file = find_latest_file(song_dir, "Gemini_onsets_*.csv")
        if gemini_file:
            g_onsets = load_onsets_csv(gemini_file)
            g_metrics = score_onsets(orig_onsets, g_onsets, DEFAULT_TOLERANCE_MS)
            gtp, gfp = g_metrics['tp'], g_metrics['fp']
        else:
            gtp, gfp = 0, 0
            
        # Qwen
        qwen_file = find_latest_file(song_dir, "Qwen_onsets_*.csv")
        if qwen_file:
            q_onsets = load_onsets_csv(qwen_file)
            q_metrics = score_onsets(orig_onsets, q_onsets, DEFAULT_TOLERANCE_MS)
            qtp, qfp = q_metrics['tp'], q_metrics['fp']
        else:
            qtp, qfp = 0, 0
            
        # Only add songs that have at least one prediction file
        if gemini_file or qwen_file:
            # truncate song names if they are too long for the X axis
            display_name = song[:25] + ".." if len(song) > 27 else song
            song_names.append(display_name)
            original_counts.append(n_orig)
            gemini_tp.append(gtp)
            gemini_fp.append(gfp)
            qwen_tp.append(qtp)
            qwen_fp.append(qfp)

    if not song_names:
        print("No prediction files found to visualize.")
        return

    print(f"Found data for {len(song_names)} songs. Generating chart...")

    # Create the stacked grouped bar chart
    x = np.arange(len(song_names))
    width = 0.35  # width of bars
    
    # Set global font size appropriately for a wide chart
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Gemini bars (Bottom = TP, Top = FP)
    bars_g_tp = ax.bar(x - width/2, gemini_tp, width, label='Gemini - Rightfully Detected (TP)', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars_g_fp = ax.bar(x - width/2, gemini_fp, width, bottom=gemini_tp, label='Gemini - Wrongfully Detected (FP)', color='#aec7e8', edgecolor='black', linewidth=0.5)
    
    # Qwen bars (Bottom = TP, Top = FP)
    bars_q_tp = ax.bar(x + width/2, qwen_tp, width, label='Qwen - Rightfully Detected (TP)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    bars_q_fp = ax.bar(x + width/2, qwen_fp, width, bottom=qwen_tp, label='Qwen - Wrongfully Detected (FP)', color='#98df8a', edgecolor='black', linewidth=0.5)
    
    # Original Onsets counts (as a scatter marker)
    scat = ax.scatter(x, original_counts, color='red', marker='X', s=120, linewidths=1.5, edgecolors='black', label='Original Onsets (Ground Truth Total)', zorder=5)

    ax.set_ylabel('Number of Onsets (Detected)', fontsize=14, fontweight='bold')
    ax.set_title('Onset Detection Comparison: Gemini vs Qwen across Fraxtil Dataset', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(song_names, rotation=45, ha='right', fontsize=10)
    
    # Make a clean legend outside the main plot area if possible, or top right
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=11, framealpha=0.9)
    
    # Add subtle grid lines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Automatically adjust layout to fit labels nicely
    plt.tight_layout()
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onset_detection_comparison.png')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)  # Free up memory
    print(f"✅ Visualization perfectly saved out to:\n  {out_file}")

if __name__ == '__main__':
    main()
