import os
import sys
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv

# Re-use helpers from visualize_alignment_01.py if possible, or copy them over to be robust
def load_step_times_from_csv(csv_path: str) -> np.ndarray:
    times = []
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            t_idx = header.index("time_ms")
            n_idx = header.index("notes")
        except ValueError:
            return np.array([])
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(t_idx, n_idx): continue
            note = parts[n_idx].strip()
            if note in ("", "0000", ","): continue
            try:
                times.append(float(parts[t_idx]) / 1000.0)
            except ValueError:
                continue
    return np.array(sorted(times))

def nearest_distance(step_times: np.ndarray, ref_times: np.ndarray) -> np.ndarray:
    if len(ref_times) == 0: return np.full(len(step_times), np.inf)
    distances = []
    for t in step_times:
        idx = np.searchsorted(ref_times, t)
        d = np.inf
        if idx > 0: d = min(d, abs(t - ref_times[idx - 1]))
        if idx < len(ref_times): d = min(d, abs(t - ref_times[idx]))
        distances.append(d)
    return np.array(distances)

def alignment_pct(distances: np.ndarray, tol_ms: float = 50.0) -> float:
    if len(distances) == 0: return 0.0
    return (distances <= tol_ms / 1000.0).sum() / len(distances) * 100.0

BASE_DIR = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
DIFFICULTIES = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]

def find_csv(song_dir, diff):
    # Pattern: *_{diff}_*gemini-pro-latest*.csv
    files = glob.glob(os.path.join(song_dir, f"*_{diff}_*gemini-pro-latest*.csv"))
    if diff == "Challenge":
        files += glob.glob(os.path.join(song_dir, f"*_Challenging_*gemini-pro-latest*.csv"))
    # Skip _sorted.csv 
    valid = [f for f in files if not f.endswith("_sorted.csv")]
    if valid:
        # Get the newest one
        valid.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return valid[0]
    return None

def parse_sm_metadata(sm_file: str):
    """Parses the reference .sm or .ssc file to get BPMS and OFFSET."""
    metadata = {'bpm': None, 'offset': 0.0}
    try:
        with open(sm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("#OFFSET:"):
                    try:
                        val = line.split(":")[1].split(";")[0].strip()
                        metadata['offset'] = float(val)
                    except ValueError:
                        pass
                if line.strip().startswith("#BPMS:"):
                    try:
                        val = line.split(":")[1].split(";")[0].strip()
                        first_bpm = val.split(",")[0]
                        bpm_val = first_bpm.split("=")[1]
                        metadata['bpm'] = float(bpm_val)
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass
    return metadata

def build_sm_beat_grid(bpm: float, offset: float, duration: float, subdivisions_per_beat: int = 4) -> np.ndarray:
    """
    Builds a mathematical grid of target step times based on BPM and Offset.
    subdivisions_per_beat=4 means 16th notes (4 lines per beat).
    """
    if not bpm or bpm <= 0:
        return np.array([])
        
    beat_duration = 60.0 / bpm
    tick_duration = beat_duration / subdivisions_per_beat
    
    # Grid starts at -offset
    grid_times = []
    current_time = -offset
    
    # We allow some pre-start beats just in case, but usually start at 0
    if current_time < 0:
        # Fast forward to 0 or first positive tick
        while current_time < 0:
            current_time += tick_duration
            
    while current_time <= duration:
        grid_times.append(current_time)
        current_time += tick_duration
        
    return np.array(grid_times)

def load_existing_csv(csv_path: str) -> dict:
    """Loads existing CSV to allow users to manually specify files."""
    existing_files = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song = row.get('Song Name')
                diff = row.get('Difficulty')
                file_used = row.get('File_Used')
                
                if song and diff and file_used and file_used != "No file found" and file_used != "-":
                    if song not in existing_files:
                        existing_files[song] = {}
                    existing_files[song][diff] = file_used
    return existing_files

def main():
    songs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith(".ogg") or f.endswith(".mp3"):
                songs.append((os.path.basename(root), os.path.join(root, f)))
    songs.sort(key=lambda x: x[0])
    # Limit to first 20 songs 
    songs = songs[:20]

    csv_out_path = '/Users/mukeshguntumadugu/llm_beatmap_generator/gemini_pro_alignment_results.csv'
    existing_files = load_existing_csv(csv_out_path)

    results_onset = {diff: [] for diff in DIFFICULTIES}
    results_beat = {diff: [] for diff in DIFFICULTIES}
    files_used = {diff: [] for diff in DIFFICULTIES}
    song_names = []

    print("Extracting Onset Alignment Data (this may take a couple minutes)...", flush=True)

    for idx, (song_name, audio_path) in enumerate(songs):
        print(f"Processing ({idx+1}/{len(songs)}): {song_name}...", end=" ", flush=True)
        song_names.append(song_name)
        
        # Load audio once
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate Onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Calculate Beats (Old librosa method)
        # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Calculate BPM Grid (StepMania method)
        sm_file = None
        for ext in ['.sm', '.ssc']:
            potential_sm = os.path.join(os.path.dirname(audio_path), song_name + ext)
            if os.path.exists(potential_sm):
                sm_file = potential_sm
                break
                
        beat_grid = np.array([])
        if sm_file:
            print("[SM]", end=" ")
            metadata = parse_sm_metadata(sm_file)
            if metadata['bpm']:
                 beat_grid = build_sm_beat_grid(metadata['bpm'], metadata['offset'], duration, subdivisions_per_beat=4)
                 
        for diff in DIFFICULTIES:
            # First check if the user manually specified a file in the CSV
            manual_file = existing_files.get(song_name, {}).get(diff)
            csv_path = None
            
            if manual_file:
                 potential_path = os.path.join(os.path.dirname(audio_path), manual_file)
                 if os.path.exists(potential_path):
                     csv_path = potential_path
                     print(f" [Using manual file for {diff}]", end="")
                 else:
                     # Fallback to finding it automatically if manual file doesn't exist
                     csv_path = find_csv(os.path.dirname(audio_path), diff)
            else:
                 csv_path = find_csv(os.path.dirname(audio_path), diff)

            if csv_path:
                step_times = load_step_times_from_csv(csv_path)
                valid_steps = step_times[step_times <= duration]
                
                # 1. Onset Alignment
                d_onset = nearest_distance(valid_steps, onset_times)
                pct_onset = alignment_pct(d_onset, 50.0)
                results_onset[diff].append(pct_onset)
                
                # 2. Beat Alignment
                if len(beat_grid) > 0:
                    d_beat = nearest_distance(valid_steps, beat_grid)
                    pct_beat = alignment_pct(d_beat, 50.0)
                    results_beat[diff].append(pct_beat)
                else:
                    results_beat[diff].append(np.nan)
                
                files_used[diff].append(os.path.basename(csv_path))
            else:
                results_onset[diff].append(np.nan)
                results_beat[diff].append(np.nan)
                files_used[diff].append("No file found")
        print(" Done.", flush=True)
                
    print("\n--- ONSET ALIGNMENT RESULTS ---\n")
    print("| Index | Song | Beginner | Easy | Medium | Hard | Challenge |")
    print("|---|---|---|---|---|---|---|")
    for i, song in enumerate(song_names):
        row = [str(i+1), song]
        for diff in DIFFICULTIES:
            val = results_onset[diff][i]
            filename = files_used[diff][i]
            
            if np.isnan(val):
                row.append("-")
            else:
                row.append(f"{val:.1f}%<br/><sub>{filename}</sub>")
        print("| " + " | ".join(row) + " |")

    # Output to CSV file (Onset)
    with open(csv_out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Song Index', 'Song Name', 'Difficulty', 'Onset_Alignment_Pct', 'Beat_Alignment_Pct', 'File_Used'])
        for i, song in enumerate(song_names):
            for diff in DIFFICULTIES:
                val_onset = results_onset[diff][i]
                val_beat = results_beat[diff][i]
                filename = files_used[diff][i]
                
                onset_str = f"{val_onset:.1f}%" if not np.isnan(val_onset) else "-"
                beat_str = f"{val_beat:.1f}%" if not np.isnan(val_beat) else "-"
                writer.writerow([i+1, song, diff, onset_str, beat_str, filename])
    
    print(f"\nResults saved to CSV: {csv_out_path}")
        
    # --- CHART 1: ONSET ALIGNMENT ---
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    x_indices = np.arange(1, len(songs) + 1)
    
    for i, diff in enumerate(DIFFICULTIES):
        y_vals = results_onset[diff]
        x_plot = [x for x, y in zip(x_indices, y_vals) if not np.isnan(y)]
        y_plot = [y for y in y_vals if not np.isnan(y)]
        plt.scatter(x_plot, y_plot, label=diff, color=colors[i], marker=markers[i], s=120, alpha=0.8, edgecolors='w')
        
    plt.xticks(x_indices, [str(i) for i in x_indices], rotation=0)
    plt.xlabel('Song Index')
    plt.ylabel('Onset Alignment (%)')
    plt.title('Gemini Pro Latest: Onset Alignment by Difficulty (50ms tolerance)\nFirst 20 Songs in Fraxtil Database')
    plt.ylim(-5, 105) 
    plt.legend(title='Difficulty', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_path = '/Users/mukeshguntumadugu/llm_beatmap_generator/gemini_pro_onset_alignment_chart.png'
    plt.savefig(chart_path, dpi=150)
    print(f"\nOnset Chart saved to {chart_path}")
    
    # --- CHART 2: BEAT ALIGNMENT ---
    plt.figure(figsize=(14, 8))
    for i, diff in enumerate(DIFFICULTIES):
        y_vals = results_beat[diff]
        x_plot = [x for x, y in zip(x_indices, y_vals) if not np.isnan(y)]
        y_plot = [y for y in y_vals if not np.isnan(y)]
        plt.scatter(x_plot, y_plot, label=diff, color=colors[i], marker=markers[i], s=120, alpha=0.8, edgecolors='w')
        
    plt.xticks(x_indices, [str(i) for i in x_indices], rotation=0)
    plt.xlabel('Song Index')
    plt.ylabel('Beat Alignment (%)')
    plt.title('Gemini Pro Latest: Beat Alignment (BPM) by Difficulty (50ms tolerance)\nFirst 20 Songs in Fraxtil Database')
    plt.ylim(-5, 105) 
    plt.legend(title='Difficulty', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    beat_chart_path = '/Users/mukeshguntumadugu/llm_beatmap_generator/gemini_pro_beat_alignment_chart.png'
    plt.savefig(beat_chart_path, dpi=150)
    print(f"Beat Chart saved to {beat_chart_path}")

if __name__ == "__main__":
    main()
