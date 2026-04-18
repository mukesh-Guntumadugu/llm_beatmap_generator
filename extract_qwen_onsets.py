"""
extract_qwen_onsets.py
======================
Sends each of the 20 Fraxtil songs to the Qwen2-Audio model and asks it to
predict audio onset times in milliseconds. Saves results to CSV files.

Output filename: Qwen_onsets_<SongName>_<ddmmyyyyHHMMSS>.csv
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
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

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
    Returns the bulletproof anti-hallucination prompt for Qwen2-Audio.
    """
    prompt = (
        f"<Audio>\nPlease listen to the audio and extract the specific musical burst onsets from this {round(duration_sec, 2)}-second clip. "
        "You must follow these strict rules to prevent hallucinations:\n"
        "1. Do not give me a bare list of numbers. Format each line exactly as: [Hit] : [Onset Timestamp in seconds]\n"
        "2. CRITICAL: Do not interpolate, guess, auto-fill, or count up sequentially. Only output the exact numbers found in the audio.\n"
        "3. Do not make up extra data if the audio is sparse. Stop generating when you reach the end."
    )
    return prompt

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_audio_file(song_dir: str) -> Optional[str]:
    """Return the first .ogg/.mp3/.wav found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None


def parse_onsets_from_response(response_text: str, duration_sec: float) -> List[float]:
    """
    Extract onset times in milliseconds from Qwen's response.

    Handles three output formats (tried in order):
      1. JSON objects with a 'time_ms' key  (e.g. {"time_ms": 500.0, ...})
      2. A bare/JSON array of numbers       (e.g. [0, 125, 250, 500])
      3. Numbers scattered in plain text    (fallback regex scan)

    Values outside [0, 600_000] ms (0 – 10 min) are discarded as noise.
    """
    import json

    text = response_text.strip()
    text_clean = re.sub(r"```[\w]*\n?", "", text).strip()
    onsets: List[float] = []

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

    # ── Strategy 1: JSON objects with time_ms key ─────────────────────────────
    try:
        json_objects = re.findall(r'\{[^{}]+\}', text_clean)
        for obj_str in json_objects:
            try:
                obj = json.loads(obj_str)
                if 'time_ms' in obj:
                    ms = float(obj['time_ms'])
                    if 0.0 <= ms <= 600_000:
                        onsets.append(round(ms, 2))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        if onsets:
            onsets = sorted(set(onsets))
            if onsets and max(onsets) <= duration_sec * 1.5:
                # Qwen outputted seconds instead of milliseconds. Auto-scale it.
                onsets = [round(ms * 1000, 2) for ms in onsets]
            return onsets
    except Exception:
        pass

    # ── Strategy 2: JSON / bare array of numbers ──────────────────────────────
    try:
        match = re.search(r'\[([\d.,\s]+)\]', text_clean)
        if match:
            arr_str = '[' + match.group(1) + ']'
            arr = json.loads(arr_str)
            for val in arr:
                ms = float(val)
                if 0.0 <= ms <= 600_000:
                    onsets.append(round(ms, 2))
            if onsets:
                onsets = sorted(set(onsets))
                if onsets and max(onsets) <= duration_sec * 1.5:
                    onsets = [round(ms * 1000, 2) for ms in onsets]
                return onsets
    except Exception:
        pass

    # ── Strategy 3: Regex scan for bare numbers (fallback) ────────────────────
    numbers = re.findall(r'\b(\d+(?:[.,]\d+)?)\b', text_clean)
    for n in numbers:
        n = n.replace(',', '.')
        try:
            ms = float(n)
            if 0.0 <= ms <= 600_000:
                onsets.append(round(ms, 2))
        except ValueError:
            pass

    onsets = sorted(set(onsets))
    if onsets and max(onsets) <= duration_sec * 1.5:
        onsets = [round(ms * 1000, 2) for ms in onsets]
        
    return onsets



def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str) -> str:
    """Save onsets to a CSV file and return the file path."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"Qwen_onsets_{safe_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])

    return filepath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load Qwen model once before the loop
    try:
        setup_qwen()
    except Exception as e:
        print(f"ERROR: Failed to load Qwen model: {e}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_") and not d.startswith(".")
    ])

    print(f"\nFound {len(song_dirs)} song folders. Starting Qwen onset detection...\n")
    print(f"{'Song':<45} {'# Qwen Onsets':>14}  {'Output file'}")
    print("─" * 110)

    total_songs = 0

    for i, song_name in enumerate(song_dirs):
        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        if audio_path is None:
            print(f"  WARNING: [{i+1}/{len(song_dirs)}] No audio in: {song_name}")
            continue

        print(f"  [{i+1}/{len(song_dirs)}] Processing: {song_name} ...", end="", flush=True)

        max_retries = 3
        onset_ms = []
        response = ""
        duration_sec = 0.0

        for attempt in range(max_retries):
            try:
                duration_sec = librosa.get_duration(path=audio_path)
                prompt = build_onset_prompt(duration_sec)

                response = generate_beatmap_with_qwen(audio_path, prompt=prompt)

                if not response or not response.strip():
                    print(f"\n  WARNING: Empty response for '{song_name}'")
                    continue

                onset_ms = parse_onsets_from_response(response, duration_sec)

                if onset_ms and is_hallucinated_loop(onset_ms):
                    print(f"\n  [!] HALLUCINATION CAUGHT: Neural counting drift detected! Retrying...")
                    onset_ms = []
                    continue

                if onset_ms:
                    break

            except Exception as e:
                print(f"\n  ERROR processing '{song_name}': {e}")
                if attempt == max_retries - 1:
                    break

        if not onset_ms:
            print(f"\n  WARNING: No parseable onsets in response for '{song_name}'")
            raw_path = os.path.join(song_dir, f"Qwen_onsets_RAW_{song_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"     Raw response saved to: {raw_path}")
            continue

        out_path = save_onsets_csv(onset_ms, song_name, song_dir)
        rel_out = os.path.relpath(out_path, BASE_DIR)
        print(f"  -> {song_name:<43} {len(onset_ms):>14,}  {rel_out}")
        total_songs += 1

        time.sleep(1)

    print("─" * 110)
    print(f"\n  Completed {total_songs}/{len(song_dirs)} songs.\n")


if __name__ == "__main__":
    main()
