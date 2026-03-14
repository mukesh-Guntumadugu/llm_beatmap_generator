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

# Ensure project root is on the path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_onset_prompt(duration_sec: float) -> str:
    """
    Returns a concise prompt that asks Qwen to list onset times in milliseconds.
    Instructs it to output ONLY numbers — one per line — so parsing is reliable.
    """
    return (
        f"The audio is {duration_sec:.1f} seconds long.\n\n"
        "Your task: listen to the audio and identify every significant musical onset "
        "(transients, drum hits, melodic note starts, rhythmic events).\n\n"
        "Output ONLY a list of onset times in milliseconds (ms), one value per line. "
        "Do NOT include any text, explanations, headers, or units — just plain numbers.\n\n"
        "Example output:\n"
        "120\n"
        "480\n"
        "960\n"
        "1440\n"
        "...\n\n"
        "Now generate the complete list for the entire song from start to finish. "
        "DO NOT STOP EARLY."
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_audio_file(song_dir: str) -> str | None:
    """Return the first .ogg/.mp3/.wav found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None


def parse_onsets_from_response(response_text: str) -> list[float]:
    """
    Extract numeric values from the model's response.
    Accepts integers or floats, ignoring any surrounding text.
    """
    # Match any standalone number (int or float), possibly with comma/dot as decimal
    numbers = re.findall(r"\b(\d+(?:[.,]\d+)?)\b", response_text)
    onsets = []
    for n in numbers:
        n = n.replace(",", ".")  # normalise European decimal separator
        try:
            ms = float(n)
            # Sanity check: ignore values that seem like year numbers, etc.
            if 0.0 <= ms <= 600_000:   # ≤ 10 minutes in ms
                onsets.append(round(ms, 2))
        except ValueError:
            pass
    return sorted(set(onsets))   # deduplicate and sort


def save_onsets_csv(onset_ms: list[float], song_name: str, out_dir: str) -> str:
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
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load Qwen model once before the loop
    try:
        setup_qwen()
    except Exception as e:
        print(f"❌ Failed to load Qwen model: {e}")
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
            print(f"  ⚠️  [{i+1}/{len(song_dirs)}] No audio in: {song_name}")
            continue

        print(f"  [{i+1}/{len(song_dirs)}] Processing: {song_name} ...", end="", flush=True)

        try:
            # Get song duration for the prompt
            duration_sec = librosa.get_duration(path=audio_path)
            prompt = build_onset_prompt(duration_sec)

            response = generate_beatmap_with_qwen(audio_path, prompt=prompt)

            if not response or not response.strip():
                print(f"\n  ⚠️  Empty response for '{song_name}'")
                continue

            onset_ms = parse_onsets_from_response(response)

            if not onset_ms:
                print(f"\n  ⚠️  No parseable onsets in response for '{song_name}'")
                # Save raw response for debugging
                raw_path = os.path.join(song_dir, f"Qwen_onsets_RAW_{song_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"     Raw response saved to: {raw_path}")
                continue

            out_path = save_onsets_csv(onset_ms, song_name, song_dir)
            rel_out = os.path.relpath(out_path, BASE_DIR)
            print(f"  ✅ {song_name:<43} {len(onset_ms):>14,}  {rel_out}")
            total_songs += 1

        except Exception as e:
            print(f"\n  ❌ Error processing '{song_name}': {e}")

        time.sleep(1)   # brief pause between songs

    print("─" * 110)
    print(f"\n✅  Completed {total_songs}/{len(song_dirs)} songs.\n")


if __name__ == "__main__":
    main()
