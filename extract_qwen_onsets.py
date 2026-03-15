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

def build_onset_prompt(duration_sec: float) -> str:
    """
    Returns the onset detection prompt for Qwen2-Audio.
    Uses the same detailed instruction as Gemini for fair model comparison.
    """
    system_instruction = (
        "You are an expert music analyst and audio engineer specializing in precise "
        "onset detection for rhythm game chart generation.\n\n"

        "## What is an Onset?\n"
        "An onset is the exact moment a new musical event begins — the attack phase "
        "of a sound. Onsets occur on:\n"
        "- Percussion: kick drum, snare, hi-hat, cymbal, clap, tom hits\n"
        "- Melodic instruments: guitar pick attack, piano key strike, bass pluck, "
        "synth note start, violin bow attack\n"
        "- Vocals: consonant or vowel attacks at the start of sung words or syllables\n"
        "- Any transient: any sudden increase in energy that marks the start of a "
        "rhythmic or melodic event\n\n"

        "## Your Task\n"
        "When given an audio file, you must:\n"
        "1. Listen to the complete audio from the very beginning to the very end. "
        "Do not stop early.\n"
        "2. Identify every significant musical onset throughout the entire duration.\n"
        "3. Record the exact time of each onset in milliseconds (ms), measured from "
        "the start of the audio (time 0).\n"
        "4. Return all onset times as a single JSON array of numbers.\n\n"

        "## Detection Guidelines\n"
        "- Be thorough: a typical 3-minute song should have hundreds of onsets.\n"
        "- Be precise: onset times should be accurate to within ±5 milliseconds.\n"
        "- Include ALL instrument layers: if a kick drum and a hi-hat hit at the same "
        "time, record that time once (it is one onset event).\n"
        "- Include weak onsets: even soft notes or ghost notes on a snare should be "
        "captured if they are rhythmically significant.\n"
        "- Do not hallucinate: only report onsets you can actually hear in the audio. "
        "Do not invent onsets where there is silence.\n"
        "- Cover the full song: make sure the last few seconds of the song are "
        "included — many submissions fail by stopping too early.\n\n"

        "## Output Format\n"
        "You MUST output ONLY a valid JSON array of numbers, nothing else.\n"
        "- Each number is an onset time in milliseconds (integer or float).\n"
        "- The array must be sorted in ascending order (earliest onset first).\n"
        "- Do NOT include any explanation, markdown formatting, headers, units, "
        "or any text outside the JSON array.\n"
        "- Do NOT wrap the array in backticks or code fences.\n"
        "- Correct format: [0, 125.5, 250, 375, 500, 750.25, 1000, ...]\n"
        "- Incorrect formats:\n"
        "    'Here are the onsets: [0, 125, 250]'  ← has explanation text\n"
        "    '```json\\n[0, 125, 250]\\n```'         ← has markdown fencing\n"
        "    '{\"onsets\": [0, 125, 250]}'           ← wrong structure\n\n"

        "## Quality Criteria\n"
        "Your output will be evaluated against a ground-truth onset list generated "
        "by a professional audio analysis tool (librosa). A good onset detection "
        "result achieves:\n"
        "- Precision ≥ 60%: most of your predicted onsets should match real onsets\n"
        "- Recall ≥ 60%: you should find at least 60% of the real onsets\n"
        "- F1 Score ≥ 0.60: the harmonic mean of precision and recall\n"
        "A predicted onset counts as correct if it is within ±50 ms of a "
        "ground-truth onset.\n\n"

        "Remember: output ONLY the JSON array. No other text."
    )
    per_song = (
        f"The audio is {duration_sec:.1f} seconds long. "
        "Identify all musical onsets and return them as a JSON array of "
        "millisecond timestamps covering the full duration."
    )
    return system_instruction + "\n\n" + per_song

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
    # Strip markdown code fences (```json ... ``` etc.)
    text_clean = re.sub(r"```[\w]*\n?", "", text).strip()

    onsets: List[float] = []

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

            onset_ms = parse_onsets_from_response(response, duration_sec)

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
