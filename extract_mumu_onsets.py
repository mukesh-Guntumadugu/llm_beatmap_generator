"""
extract_mumu_onsets.py
======================
Sends each of the 20 Fraxtil songs to the MuMu-LLaMA model and asks it to
predict audio onset times in milliseconds. Saves results to CSV files.

Output filename: Mumu_onsets_<SongName>_<ddmmyyyyHHMMSS>.csv
Output columns : onset_index, onset_ms
"""

import os
import re
import csv
import sys
import time
import datetime
import librosa
from typing import Optional, List

# Ensure project root is on the path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# ── Prompt Builder ────────────────────────────────────────────────────────────

def is_hallucinated_loop(onsets: List[float]) -> bool:
    if not onsets or len(onsets) < 10:
        return False
    if len(onsets) > 150:
        return True 
    times = onsets
    deltas = [round(times[j+1] - times[j], 1) for j in range(len(times)-1)]
    for j in range(len(deltas) - 8):
        window = deltas[j:j+8]
        if all(w == window[0] for w in window):
            return True
    return False

def build_onset_prompt(duration_sec: float) -> str:
    """
    Returns the bulletproof anti-hallucination prompt.
    """
    prompt = (
        f"<Audio>\nPlease listen to the audio and extract the specific musical burst onsets from this {round(duration_sec, 2)}-second clip. "
        "You must follow these strict rules to prevent hallucinations:\n"
        "1. Do not give me a bare list of numbers. Format each line exactly as: [Hit] : [Onset Timestamp in seconds]\n"
        "2. CRITICAL: Do not interpolate, guess, auto-fill, or count up sequentially. Only output the exact numbers found in the audio.\n"
        "3. Do not make up extra data if the audio is sparse. Stop generating when you reach the end."
    )
    return prompt


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_onsets_from_response(text: str, duration_sec: float) -> Optional[List[float]]:
    if not text:
        return None

    text_clean = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_clean, re.DOTALL | re.IGNORECASE)
    if match:
        text_clean = match.group(1).strip()

    onsets = []
    
    # 1. Anti-Hallucination Secure Anchored Parsing (Seconds)
    found_matches = re.findall(r':\s*([0-9]+\.?[0-9]*)', text_clean)
    if found_matches:
        for val in found_matches:
            try:
                ms = float(val) * 1000.0
                if 0.0 <= ms <= 600_000:
                    onsets.append(round(ms, 2))
            except ValueError:
                pass
        if onsets:
            return sorted(set(onsets))

    # 2. Extract just the array portion `[...]`
    match = re.search(r"\[(.*?)\]", text_clean, re.DOTALL)
    if not match:
        return None

    array_str = match.group(1)
    
    try:
        raw_vals = [float(x.strip()) for x in array_str.split(',') if x.strip()]
        if not raw_vals:
            return None
            
        max_val = max(raw_vals)
        if max_val > 0 and max_val <= (duration_sec * 1.5): # e.g. 180s vs 180,000ms
            print("  [Auto-correction: Converted MuMu seconds output to milliseconds]")
            raw_vals = [v * 1000 for v in raw_vals]

        return sorted(raw_vals)
    except Exception as parse_err:
        print(f"  WARNING: Parse error: {parse_err}")
        return None

# ── File / OS Helpers ─────────────────────────────────────────────────────────

def find_audio_file(folder_path: str) -> Optional[str]:
    """Finds the first .ogg, .mp3, or .wav in a folder."""
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._"):
            return os.path.join(folder_path, f)
    return None

def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str) -> str:
    """Save onsets to a CSV file and return the file path."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"Mumu_onsets_{safe_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])

    return filepath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract onsets using MuMu-LLaMA.")
    parser.add_argument("--start", type=int, default=1, help="Song index to start from (1-based)")
    parser.add_argument("--end", type=int, default=20, help="Song index to end at (inclusive)")
    args = parser.parse_args()

    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load MuMu model once before the loop
    try:
        setup_mumu()
    except Exception as e:
        print(f"ERROR: Failed to setup MuMu-LLaMA interface: {e}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_") and not d.startswith(".")
    ])

    print(f"\nProcessing songs {args.start} to {args.end} using MuMu-LLaMA...\n")
    print(f"{'Song':<45} {'# Mumu Onsets':>14}  {'Output file'}")
    print("─" * 110)

    for i, song_name in enumerate(song_dirs):
        song_idx = i + 1
        if not (args.start <= song_idx <= args.end):
            continue

        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        if audio_path is None:
            print(f"  WARNING: [{song_idx}/{len(song_dirs)}] No audio in: {song_name}")
            continue

        print(f"  [{song_idx}/{len(song_dirs)}] Processing: {song_name} ...", end="", flush=True)

        max_retries = 3
        onset_ms = []
        response = ""
        duration_sec = 0.0

        for attempt in range(max_retries):
            try:
                # Get song duration for the prompt
                duration_sec = librosa.get_duration(path=audio_path)
                prompt = build_onset_prompt(duration_sec)

                response = generate_beatmap_with_mumu(audio_path, prompt=prompt)

                if not response or not response.strip():
                    print(f"\n  WARNING: Empty response for '{song_name}'")
                    continue

                onset_ms = parse_onsets_from_response(response, duration_sec)

                if onset_ms and is_hallucinated_loop(onset_ms):
                    print("\n  [!] HALLUCINATION CAUGHT: Neural counting drift detected! Retrying...")
                    onset_ms = []
                    continue

                if onset_ms:
                    break # Success!

            except Exception as e:
                print(f"\n  ERROR processing '{song_name}': {e}")
                if attempt == max_retries - 1:
                    break

        if not onset_ms:
            print(f"\n  WARNING: No parseable valid onsets in response for '{song_name}' after {max_retries} attempts.")
            # Save raw response for debugging
            raw_path = os.path.join(song_dir, f"Mumu_onsets_RAW_{song_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response)
            continue

        # Save valid onsets to CSV
        csv_path = save_onsets_csv(onset_ms, song_name, song_dir)
        
        print(f"\r{' ' * 80}\r", end="")
        print(f"{song_name[:43]:<45} {len(onset_ms):>14d}  {os.path.basename(csv_path)}")

    print("\nMuMu extraction run completed.")

if __name__ == "__main__":
    main()
