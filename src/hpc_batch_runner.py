"""
HPC Batch Runner — queries the local Qwen2-Audio server (no API keys).

Usage:
    # Make sure the server is running first:
    #   python src/hpc_qwen_server.py
    
    python src/hpc_batch_runner.py                  # new task
    python src/hpc_batch_runner.py 1                # resume task 1
    python src/hpc_batch_runner.py --server http://localhost:8000  # custom server URL
"""

import os
import sys

# ── Ensure project root is on sys.path when run as `python3 src/hpc_batch_runner.py` ──
# Running the script directly puts `src/` on sys.path, making `from src.beatmap_prompt`
# resolve to `src/src/beatmap_prompt.py` (wrong). We fix that here.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import csv
import json
import base64
import datetime
import argparse
import requests
import librosa

from src.beatmap_prompt import build_qwen_prompt
import re


DIFFICULTIES = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]
MODEL_NAME   = "Qwen2-Audio-7B"
SERVER_URL   = "http://localhost:8000"

# Where the Fraxtil songs live (relative to project root)
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# Task registry (same format as batch_process_fraxtil.py)
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".task_registry_hpc.json")

# ── Prompt ────────────────────────────────────────────────────────────────────
def parse_sm_metadata(sm_file: str) -> float:
    """Parses the reference .sm or .ssc file to get the base BPM."""
    try:
        with open(sm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("#BPMS:"):
                    # Format: #BPMS:0.000=150.000;
                    val = line.split(":")[1].split(";")[0].strip()
                    first_bpm = val.split(",")[0]
                    bpm_val = first_bpm.split("=")[1]
                    return float(bpm_val)
    except Exception:
        pass
    return None

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(duration: float, difficulty: str = "Medium", bpm: float = None) -> str:
    """Uses the shared prompt (same as Gemini) from src/beatmap_prompt.py."""
    return build_qwen_prompt(difficulty, duration, bpm)

# ── Robust Qwen output parser ─────────────────────────────────────────────────
def _parse_qwen_output(text: str) -> list[dict]:
    """
    Handles all the ways Qwen might format its response:
      1. Clean JSON array: [{...}, ...]
      2. JSON objects wrapped in a markdown code block (```json ... ```)
      3. JSON objects with preamble/explanation text before them
      4. Plain 4-char note rows (fallback)

    Always returns a list of dicts with keys:
      time_ms, beat_position, notes, placement_type, note_type, confidence, instrument
    """
    # ── Step 1: Strip markdown code fences (```json, ```css, ``` etc.) ────────
    clean = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")

    # ── Step 2: Try to extract a JSON array [...] from anywhere in the text ───
    array_match = re.search(r"\[.*\]", clean, re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list) and data:
                rows = []
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "notes" in item:
                        rows.append({
                            "time_ms":        float(item.get("time_ms", i * 125.0)),
                            "beat_position":   float(item.get("beat_position", round(i / 4.0, 3))),
                            "notes":           str(item["notes"]),
                            "placement_type":  int(item.get("placement_type", 1)),
                            "note_type":       int(item.get("note_type", 2)),
                            "confidence":      float(item.get("confidence", 0.9)),
                            "instrument":      str(item.get("instrument", "mixed")),
                        })
                if rows:
                    print(f"  ✅ Parsed {len(rows)} rows from JSON array.")
                    return rows
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Step 3: Try to extract individual JSON objects {...} line by line ──────
    rows = []
    for i, line in enumerate(clean.splitlines()):
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                item = json.loads(line)
                if "notes" in item:
                    rows.append({
                        "time_ms":        float(item.get("time_ms", i * 125.0)),
                        "beat_position":   float(item.get("beat_position", round(i / 4.0, 3))),
                        "notes":           str(item["notes"]),
                        "placement_type":  int(item.get("placement_type", 1)),
                        "note_type":       int(item.get("note_type", 2)),
                        "confidence":      float(item.get("confidence", 0.9)),
                        "instrument":      str(item.get("instrument", "mixed")),
                    })
            except (json.JSONDecodeError, ValueError):
                pass
    if rows:
        print(f"  ✅ Parsed {len(rows)} rows from individual JSON objects.")
        return rows

    # ── Step 4: Fallback — plain 4-character note rows ─────────────────────────
    fallback = []
    rows_per_sec = 8.0
    beat_idx = 1.0
    row_count = 0
    for line in clean.splitlines():
        note = line.strip()
        if note == ",":
            beat_idx += 1.0
        elif len(note) == 4 and all(c in "01234M" for c in note):
            fallback.append({
                "time_ms":        round(row_count / rows_per_sec * 1000, 2),
                "beat_position":   round(beat_idx + (row_count % int(rows_per_sec)) / rows_per_sec, 3),
                "notes":           note,
                "placement_type":  1,
                "note_type":       2,
                "confidence":      0.9,
                "instrument":      "mixed",
            })
            row_count += 1
    if fallback:
        print(f"  ✅ Parsed {len(fallback)} rows from plain note lines (fallback).")
        return fallback

    print("  ⚠️  Could not parse any valid rows from Qwen output.")
    return []

# ── Task registry helpers ──────────────────────────────────────────────────────
def load_registry():
    if os.path.exists(_REGISTRY_FILE):
        try:
            with open(_REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_task_id": 0, "tasks": {}}

def save_registry(reg):
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)

def create_new_task(reg, difficulty: str) -> int:
    new_id = reg["last_task_id"] + 1
    reg["last_task_id"] = new_id
    reg["tasks"][str(new_id)] = {
        "created": datetime.datetime.now().isoformat(),
        "difficulty": difficulty,
    }
    save_registry(reg)
    return new_id

# ── Audio helpers ─────────────────────────────────────────────────────────────
def audio_to_b64(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ── Song processing ───────────────────────────────────────────────────────────
def get_target_files(base_dir):
    audio_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)

def process_song(audio_path: str, task_id: int, server_url: str, difficulty: str):
    name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    dirname     = os.path.dirname(audio_path)
    task_tag    = f"task{task_id:04d}"

    print(f"Processing: {name_no_ext}  [{difficulty}]")

    # Skip if already done for this task + difficulty
    existing = [
        f for f in os.listdir(dirname)
        if f.startswith(f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_") and f.endswith(".txt")
    ]
    if existing:
        print(f"  Skipping — already done ({task_tag}, {difficulty}): {existing[0]}")
        return

    try:
        duration = librosa.get_duration(path=audio_path)
        
        # Try to find the corresponding .sm or .ssc file for the BPM
        bpm = None
        for ext in [".ssc", ".sm"]:
            sm_file = os.path.join(dirname, name_no_ext + ext)
            if os.path.exists(sm_file):
                bpm = parse_sm_metadata(sm_file)
                break
                
        if bpm:
            print(f"  Duration: {duration:.1f}s  |  BPM: {bpm:.1f}  |  Encoding audio...")
        else:
            print(f"  Duration: {duration:.1f}s  |  Encoding audio...")

        audio_b64 = audio_to_b64(audio_path)
        prompt    = build_prompt(duration, difficulty, bpm)

        print(f"  Sending to local Qwen server ({server_url})...")
        resp = requests.post(
            f"{server_url}/generate",
            json={
                "audio_b64":      audio_b64,
                "audio_filename": os.path.basename(audio_path),
                "prompt":         prompt,
                "max_new_tokens": 8192,
            },
            timeout=600,  # 10-minute timeout per song
        )
        resp.raise_for_status()
        text = resp.json()["text"]

        if not text.strip():
            print("  No output from model.")
            return

        # ── Parse Qwen response (handles JSON, markdown, plain text) ──────
        parsed_rows = _parse_qwen_output(text)
        if not parsed_rows:
            print("  ⚠️  No valid rows parsed. Saving raw response as .txt for inspection.")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(dirname, f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_{timestamp}_RAW.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(text)
            return

        # ── Save files ────────────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base      = f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_{timestamp}"

        # Plain beatmap .txt (just the notes column, same as Gemini .txt)
        txt_path = os.path.join(dirname, f"{base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for row in parsed_rows:
                f.write(f"{row['notes']}\n")
        print(f"  Beatmap → {os.path.basename(txt_path)}")

        # Structured CSV matching Gemini output format exactly
        csv_path = os.path.join(dirname, f"{base}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_ms", "beat_position", "notes",
                             "placement_type", "note_type", "confidence", "instrument"])
            for row in parsed_rows:
                writer.writerow([
                    row["time_ms"], f"{row['beat_position']:.3f}",
                    row["notes"], row["placement_type"], row["note_type"],
                    f"{row['confidence']:.3f}", row["instrument"]
                ])
        print(f"  Full CSV → {os.path.basename(csv_path)}")

        print("  Done.")

    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot reach server at {server_url}. Is it running?")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HPC Batch Runner for Qwen2-Audio beatmap generation")
    parser.add_argument("task_id", nargs="?", type=int, help="Resume a specific task ID")
    parser.add_argument("--server", default=SERVER_URL, help="Qwen server URL")
    parser.add_argument("--song", default=None, help="Test with one song (partial name, e.g. 'Bad Ketchup')")
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=DIFFICULTIES,
        help="Single difficulty to generate (default: all five difficulties sequentially)",
    )
    args = parser.parse_args()

    # Resolve difficulty list
    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    # Check server health
    try:
        health = requests.get(f"{args.server}/health", timeout=5).json()
        if not health.get("model_loaded"):
            print("⚠️  Server is up but model is not loaded yet. Waiting...")
    except Exception:
        print(f"❌ Cannot reach Qwen server at {args.server}.")
        print("   Start it first:  python src/hpc_qwen_server.py")
        sys.exit(1)

    # Task ID
    registry = load_registry()
    if args.task_id:
        task_id = args.task_id
        if str(task_id) not in registry["tasks"]:
            print(f"Error: Task ID {task_id} not found in registry.")
            sys.exit(1)
        print(f"▶  Resuming HPC Task ID: {task_id:04d}")
    else:
        first_difficulty = difficulties[0]
        task_id = create_new_task(registry, first_difficulty)
        print(f"▶  New HPC Task ID: {task_id:04d}")

    print(f"   Difficulties : {', '.join(difficulties)}\n")

    target_files = get_target_files(BASE_DIR)

    # Single-song test mode
    if args.song:
        target_files = [f for f in target_files if args.song.lower() in os.path.basename(os.path.dirname(f)).lower()]
        if not target_files:
            print(f"\u274c No audio file found matching '{args.song}'. Check the song name.")
            sys.exit(1)
        print(f"\U0001f3b5 Single-song test: {os.path.basename(target_files[0])}\n")
    else:
        print(f"Found {len(target_files)} audio files.\n")

    for difficulty in difficulties:
        print(f"\n{'='*60}")
        print(f"  Difficulty: {difficulty}")
        print(f"{'='*60}\n")
        for i, audio_file in enumerate(target_files):
            print(f"[{i+1}/{len(target_files)}] ", end="")
            process_song(audio_file, task_id, args.server, difficulty)
            time.sleep(1)

if __name__ == "__main__":
    main()
