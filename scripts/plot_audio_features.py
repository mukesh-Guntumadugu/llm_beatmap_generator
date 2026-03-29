#!/usr/bin/env python3
import os
import sys
import json
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_better_onsets(json_path, audio_path, output_png):
    print(f"Loading features from '{json_path}'...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert ms back to sec for librosa plot alignment
    onsets_ms = data.get("onsets_ms", [])
    if not onsets_ms:
        print("Error: JSON missing onsets_ms.")
        return
    onset_times = [ms / 1000.0 for ms in onsets_ms]

    print(f"Loading raw audio from '{audio_path}' (this might take a few seconds)...")
    if not os.path.exists(audio_path):
        print(f"Error: Could not find audio file at {audio_path}")
        return
        
    y, sr = librosa.load(audio_path, sr=None)
    
    print("Calculating Onset Envelope for graphical background...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)

    # Set up matplotlib aesthetics
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.set_title(f"Precision Onset Graph: {data.get('audio_file', 'Audio')}", fontsize=18, pad=15, color='#e0e0e0', fontweight='bold')
    
    # 1. Plot the continuous audio energy (Onset Envelope)
    ax.plot(times, onset_env, label='Onset Strength (Audio Energy)', color='#33ccff', alpha=0.6, linewidth=1.5)
    ax.fill_between(times, onset_env, 0, color='#33ccff', alpha=0.1)
    
    # 2. Plot the exact Onset Hits (calculated points) directly on top
    # We find the peak y-values for each onset time so the dots sit perfectly on the spikes
    onset_frames = librosa.time_to_frames(onset_times, sr=sr)
    onset_env_peaks = [onset_env[min(f, len(onset_env)-1)] for f in onset_frames]
    
    ax.scatter(onset_times, onset_env_peaks, color='#ff00ff', s=40, zorder=5, label=f'Detected Onsets ({len(onset_times)} hits)')
    ax.vlines(onset_times, ymin=0, ymax=onset_env_peaks, colors='#ff00ff', linewidth=0.8, alpha=0.7, linestyles='dashed')

    ax.set_xlim(0, max(times))
    ax.set_ylim(0, max(onset_env) * 1.05)
    ax.set_ylabel("Normalized Energy", fontsize=12, color='#cccccc')
    ax.set_xlabel("Time (seconds)", fontsize=12, color='#cccccc')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # Zoom into a 20-second window if song is too long, to show details nicely
    # (Comment this out to show the whole song, but a zoomed version looks much "better graphically")
    ax.set_xlim(0, 30)
    ax.set_title(f"Precision Onset Graph (0-30s Detail): {data.get('audio_file', 'Audio')}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Successfully generated high-def plot at '{output_png}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot high-quality audio onset strength.")
    parser.add_argument("json_file", help="Path to input JSON features file.")
    parser.add_argument("audio_file", help="Path to raw audio file (.ogg).")
    parser.add_argument("-o", "--output", default="onset_plot.png", help="Output PNG path.")
    args = parser.parse_args()
    plot_better_onsets(args.json_file, args.audio_file, args.output)
