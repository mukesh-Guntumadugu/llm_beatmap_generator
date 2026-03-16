"""
generate_qwen_beatmap.py
========================
Reads pre-detected onset timestamps from `original_onsets_*.csv` files (produced
by librosa) and asks Qwen2-Audio to SELECT which onsets deserve a note and generate
a full 7-column BeatCSV format (time, beat, notes, etc).

The model is NOT asked to find onsets blindly — it is given the exact ms timestamps.
It must output CSV rows mimicking the Gemini beatmap format.

Output per file: 7-column CSV 
  time_ms,beat_position,notes,placement_type,note_type,confidence,instrument

Run per song × difficulty × 6 repetitions = 30 files / song, 600 total.

Usage:
    python3 generate_qwen_beatmap.py                    # all songs, all difficulties, 6 runs
    python3 generate_qwen_beatmap.py --song "Bad Ketchup" --difficulty easy --runs 1
"""

import os
import re
import csv
import sys
import glob
import time
import datetime
import argparse
from typing import List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

DIFFICULTIES = ["beginner", "easy", "medium", "hard", "challenging"]

# For each difficulty: approx % of onsets to place notes on
DENSITY = {
    "beginner":    0.25,
    "easy":        0.40,
    "medium":      0.55,
    "hard":        0.70,
    "challenging": 0.90,
}

# Human-readable guidance per difficulty (used in the prompt)
DIFFICULTY_DESC = {
    "beginner":    "very sparse — only place notes on the strongest, most obvious beats. "
                   "Leave long gaps so a new player has time to react.",
    "easy":        "sparse — focus on the main beat and obvious melodic hits. "
                   "Occasional off-beat notes are fine.",
    "medium":      "moderate density — cover the main groove and most melodic events. "
                   "Include some rapid patterns but leave natural breathing room.",
    "hard":        "dense — cover nearly all rhythmically significant onsets. "
                   "Fast passages should have rapid consecutive notes.",
    "challenging": "very dense — use almost every onset. "
                   "Very fast sequences and complex rhythm patterns are expected.",
}

VALID_WIDTHS = {4, 8, 12, 16}

# ── Onset loader ──────────────────────────────────────────────────────────────

def load_original_onsets(song_dir: str) -> Optional[List[float]]:
    """Return onset timestamps (ms) from the first original_onsets_*.csv found."""
    pattern = os.path.join(song_dir, "original_onsets_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    onsets = []
    with open(matches[-1], newline="", encoding="utf-8") as f:  # use latest
        reader = csv.DictReader(f)
        for row in reader:
            try:
                onsets.append(float(row["onset_ms"]))
            except (KeyError, ValueError):
                pass
    return sorted(onsets) if onsets else None


def find_audio_file(song_dir: str) -> Optional[str]:
    """Return the first audio file found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_beatmap_prompt(onsets_ms: List[float], difficulty: str) -> str:
    target_pct = int(DENSITY[difficulty] * 100)
    desc = DIFFICULTY_DESC[difficulty]
    onset_str = ", ".join(f"{ms:.2f}" for ms in onsets_ms)
    total = len(onsets_ms)

    return (
        f"You are a professional rhythm game chart designer creating a **{difficulty.upper()}** "
        f"difficulty beatmap for a 4-panel arrow game (like DDR/Stepmania).\n\n"

        f"## Available Onset Timestamps\n"
        f"The following {total} timestamps (in milliseconds) are musically significant moments "
        f"detected in the song. You must choose which ones deserve a note placement:\n\n"
        f"[{onset_str}]\n\n"

        f"## Your Task\n"
        f"Create a **{difficulty.upper()}** chart with approximately **{target_pct}%** of these "
        f"onsets activated ({desc}).\n\n"
        
        "=== MEASURE STRUCTURE — READ CAREFULLY ===\n"
        "A measure is a group of rows ended by a separator row (notes=',').\n"
        "EACH measure MUST contain EXACTLY 4, 8, 12, or 16 note rows before the ','.\n"
        "  - 4 rows  = quarter note grid   → 1 row per beat (sparse / slow music)\n"
        "  - 8 rows  = eighth note grid    → 2 rows per beat (moderate density)\n"
        "  - 12 rows = triplet grid        → 3 rows per beat (triplet feel)\n"
        "  - 16 rows = sixteenth note grid → 4 rows per beat (dense / fast music)\n"
        "For every subdivision slot that has NO note, you MUST output '0000'.\n\n"
        
        "=== OUTPUT FORMAT ===\n"
        "Output ONLY plain CSV rows (no header, no extra JSON wrapping, no explanations).\n"
        "Each line MUST exactly match this 7-column format:\n"
        "time_ms,beat_position,notes,placement_type,note_type,confidence,instrument\n\n"
        
        "For note rows:   time_ms,beat_position,1000,4,2,0.95,kick\n"
        "For empty rows:  time_ms,beat_position,0000,0,3,1.0,unknown\n"
        'For separators:  time_ms,beat_position,",",-1,-1,1.0,separator\n\n'
        
        "Example output (copy this format exactly):\n"
        "0.0,1.0,1000,4,2,0.95,kick\n"
        "125.0,1.25,0000,0,3,1.0,unknown\n"
        "250.0,1.5,0010,4,3,0.88,snare\n"
        "375.0,1.75,0000,0,3,1.0,unknown\n"
        "500.0,2.0,2000,4,2,1.0,bass\n"
        "625.0,2.25,0000,0,3,1.0,unknown\n"
        "750.0,2.5,0100,4,3,0.82,snare\n"
        "875.0,2.75,0000,0,3,1.0,unknown\n"
        "1000.0,3.0,3001,4,2,0.91,kick\n"
        '1125.0,3.25,",",-1,-1,1.0,separator\n\n'
        
        f"Now generate the {difficulty.upper()} CSV chart:"
    )

# ── Response parser ───────────────────────────────────────────────────────────

def parse_beatmap_response(
    response_text: str,
    valid_onsets: List[float],
    tolerance_ms: float = 50.0
) -> List[Tuple]:
    """
    Parse the 7-column CSV output from Qwen:
    time_ms,beat_pos,notes,placement,note_type,conf,inst
    """
    results = []

    for line in response_text.splitlines():
        line = line.strip()
        if not line or line.startswith('time_ms') or "```" in line:
            continue
            
        parts = line.split(',')
        # Handle the separator note which is "," inside quotes, or unquoted
        if len(parts) >= 7 and parts[2] in ('"', '","', '', ' '):
            parts[2] = ','
            # Shift parts back if they got split by the note comma
            if len(parts) > 7:
                parts = [parts[0], parts[1], ',', parts[4], parts[5], parts[6], parts[7]]
        
        if len(parts) < 7:
            continue
            
        try:
            t_ms = float(parts[0])
            beat = float(parts[1])
            notes = parts[2].strip(' "')
            place = int(parts[3])
            ntype = int(parts[4])
            conf = float(parts[5])
            inst = parts[6].strip(' "')
            
            # If it's an actual note hitting, try to snap it to a valid onset
            if notes != ',' and notes != '0000':
                nearest = min(valid_onsets, key=lambda v: abs(v - t_ms))
                if abs(nearest - t_ms) <= tolerance_ms:
                    t_ms = nearest
                    
            results.append((t_ms, beat, notes, place, ntype, conf, inst))
        except ValueError:
            continue

    return results

# ── CSV saver ─────────────────────────────────────────────────────────────────

def save_beatmap_csv(
    entries: List[Tuple],
    song_name: str,
    difficulty: str,
    run_num: int,
    out_dir: str
) -> str:
    ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe = song_name.replace(" ", "_").replace("/", "-")
    fname = f"Qwen_beatmap_{safe}_{difficulty}_run{run_num:02d}_{ts}.csv"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "beat_position", "notes", "placement_type", "note_type", "confidence", "instrument"])
        for row in entries:
            writer.writerow(row)
    return fpath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen beatmaps from librosa onsets.")
    parser.add_argument("--song", type=str, default=None,
                        help="Only process this song name (substring match)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=DIFFICULTIES,
                        help="Only process this difficulty level")
    parser.add_argument("--runs", type=int, default=6,
                        help="Number of generation runs per song × difficulty (default: 6)")
    args = parser.parse_args()

    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load Qwen model once
    try:
        setup_qwen()
    except Exception as e:
        print(f"❌ Failed to load Qwen model: {e}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and not d.startswith("_") and not d.startswith(".")
    ])

    if args.song:
        song_dirs = [s for s in song_dirs if args.song.lower() in s.lower()]

    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    total_files = 0
    total_expected = len(song_dirs) * len(difficulties) * args.runs

    print(f"\nGenerating Qwen beatmaps: {len(song_dirs)} songs × "
          f"{len(difficulties)} difficulty levels × {args.runs} runs "
          f"= {total_expected} files\n")
    print(f"{'Song':<40} {'Diff':<12} {'Run':<5} {'Notes':<8}  {'Output file'}")
    print("─" * 110)

    for song_name in song_dirs:
        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)
        onsets = load_original_onsets(song_dir)

        if onsets is None:
            print(f"  ⚠️  No original_onsets_*.csv found in: {song_name}")
            continue
        if audio_path is None:
            print(f"  ⚠️  No audio file found in: {song_name}")
            continue

        for difficulty in difficulties:
            prompt = build_beatmap_prompt(onsets, difficulty)

            for run in range(1, args.runs + 1):
                label = f"[{song_name[:35]:<35}] {difficulty:<10} run {run}/{args.runs}"
                print(f"  {label} ...", end="", flush=True)

                try:
                    response = generate_beatmap_with_qwen(audio_path, prompt=prompt)

                    if not response or not response.strip():
                        print(f"\n  ⚠️  Empty response — skipping")
                        continue

                    entries = parse_beatmap_response(response, onsets)

                    if not entries:
                        print(f"\n  ⚠️  No parseable entries — saving raw response")
                        ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
                        raw_path = os.path.join(
                            song_dir,
                            f"Qwen_beatmap_RAW_{song_name.replace(' ', '_')}"
                            f"_{difficulty}_run{run:02d}_{ts}.txt"
                        )
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(response)
                        print(f"     → {raw_path}")
                        continue

                    out_path = save_beatmap_csv(entries, song_name, difficulty, run, song_dir)
                    rel = os.path.relpath(out_path, BASE_DIR)
                    print(f"  ✅  {len(entries):>5} notes  →  {rel}")
                    total_files += 1

                except Exception as e:
                    print(f"\n  ❌  Error: {e}")

                time.sleep(0.5)

    print("─" * 110)
    print(f"\n✅  Generated {total_files}/{total_expected} beatmap files.\n")


if __name__ == "__main__":
    main()
