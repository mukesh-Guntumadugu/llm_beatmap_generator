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
import librosa
import soundfile as sf
import tempfile
import math
from typing import List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
from src.beatmap_prompt import BEATMAP_SYSTEM_INSTRUCTION, build_per_song_prompt, QWEN_OUTPUT_ADDENDUM

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

def get_audio_info(song_dir: str, audio_path: str) -> Tuple[float, float]:
    """Return the duration and primary BPM of a song."""
    duration = 0.0
    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        pass
        
    bpm = 120.0 # fallback
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ssc", ".sm")):
            try:
                with open(os.path.join(song_dir, f), "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                    m = re.search(r'#BPMS:.*?=([\d.]+)', content)
                    if m:
                        bpm = float(m.group(1))
                        break
            except Exception:
                pass
    return duration, bpm

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_beatmap_prompt(onsets_ms: List[float], difficulty: str, duration: float, bpm: float) -> str:
    target_pct = int(DENSITY[difficulty] * 100)
    desc = DIFFICULTY_DESC[difficulty]
    onset_str = ", ".join(f"{ms:.2f}" for ms in onsets_ms)
    total = len(onsets_ms)
    
    total_beats = (duration / 60.0) * bpm
    max_lines_16 = int(total_beats * 4) # 4 rows per beat in a 16-row measure
    max_lines_8 = int(total_beats * 2)  # 2 rows per beat in an 8-row measure

    # The dynamic injection part to string together with the shared Gemini prompt
    injection = (
        f"\n\n## Available Onset Timestamps\n"
        f"The following {total} timestamps (in milliseconds) are musically significant moments "
        f"detected in the song. You MUST select your note placements strictly from these exact numbers:\n"
        f"[{onset_str}]\n\n"
        
        f"## Density & Length Constraints\n"
        f"Your task is to create a **{difficulty.upper()}** chart with approximately **{target_pct}%** of these "
        f"onsets activated ({desc}).\n\n"
        f"The audio is {duration:.1f} seconds long with a BPM of {bpm:.1f}. This means the entire song is roughly {total_beats:.1f} total beats.\n"
        f"- If relying heavily on 16-row measures, you can output around {max_lines_16} total lines maximum.\n"
        f"- If relying on 8-row measures, expect around {max_lines_8} total lines.\n"
        f"Do NOT generate endless rows past the end of the song. Stop your notes roughly at {duration * 1000.0:.0f} ms!\n"
    )

    # Use the exact shared prompt base, then the song metadata, then our onset list, then the Qwen CSV out instruction
    return (
        BEATMAP_SYSTEM_INSTRUCTION
        + build_per_song_prompt(difficulty, duration, bpm)
        + injection
        + QWEN_OUTPUT_ADDENDUM
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
            
        duration, bpm = get_audio_info(song_dir, audio_path)

        for difficulty in difficulties:
            prompt = build_beatmap_prompt(onsets, difficulty, duration, bpm)

            for run in range(1, args.runs + 1):
                label = f"[{song_name[:35]:<35}] {difficulty:<10} run {run}/{args.runs}"
                print(f"  {label} ...")

                try:
                    audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
                    chunk_size_sec = 20.0
                    num_chunks = math.ceil(duration / chunk_size_sec)
                    
                    all_entries = []
                    
                    for i in range(num_chunks):
                        start_sec = i * chunk_size_sec
                        end_sec = min((i + 1) * chunk_size_sec, duration)
                        chunk_duration = end_sec - start_sec
                        
                        start_ms = start_sec * 1000.0
                        end_ms = end_sec * 1000.0
                        
                        # Filter onsets to just this chunk and make them relative
                        chunk_onsets = [ms - start_ms for ms in onsets if start_ms <= ms < end_ms]
                        if not chunk_onsets:
                            continue # Skip empty chunks
                        
                        start_sample = int(start_sec * sr)
                        end_sample = int(end_sec * sr)
                        chunk_audio = audio_data[start_sample:end_sample]

                        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                            tmp_path = tmp.name
                            
                        try:
                            sf.write(tmp_path, chunk_audio, sr)
                            
                            prompt = build_beatmap_prompt(chunk_onsets, difficulty, chunk_duration, bpm)
                            print(f"    Chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)...", end="", flush=True)

                            response = generate_beatmap_with_qwen(tmp_path, prompt=prompt)

                            if not response or not response.strip():
                                print(f" ⚠️ Empty response")
                                continue

                            chunk_entries = parse_beatmap_response(response, chunk_onsets)

                            if not chunk_entries:
                                print(f" ⚠️ No parseable entries")
                                continue
                                
                            # Offset timestamps back to global time
                            chunk_start_beat = (start_sec / 60.0) * bpm if bpm else 0.0
                            for entry in chunk_entries:
                                # entry: (time_ms, beat, notes, place, ntype, conf, inst)
                                global_ms = entry[0] + start_ms
                                global_beat = entry[1] + chunk_start_beat
                                all_entries.append((global_ms, global_beat, entry[2], entry[3], entry[4], entry[5], entry[6]))
                                
                            print(f" ✅ {len(chunk_entries)} notes")

                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    if not all_entries:
                        print(f"  ⚠️  No parseable entries across any chunks")
                        continue
                        
                    all_entries.sort(key=lambda x: x[0])  # Sort by global time_ms

                    out_path = save_beatmap_csv(all_entries, song_name, difficulty, run, song_dir)
                    rel = os.path.relpath(out_path, BASE_DIR)
                    print(f"  ✅  Total: {len(all_entries):>5} notes  →  {rel}")
                    total_files += 1

                except Exception as e:
                    print(f"\n  ❌  Error: {e}")

                time.sleep(0.5)

    print("─" * 110)
    print(f"\n✅  Generated {total_files}/{total_expected} beatmap files.\n")


if __name__ == "__main__":
    main()
