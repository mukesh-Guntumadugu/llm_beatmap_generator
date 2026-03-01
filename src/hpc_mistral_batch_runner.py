"""
HPC Mistral Batch Runner — queries the local Mistral-7B server.

Mirrors hpc_batch_runner.py (Qwen) but targets the Mistral server on port 8001.
Sends each audio file to the server, receives a beatmap JSON response,
and saves both a plain .txt beatmap and a full structured .csv file.

Usage:
    # Make sure the server is running first:
    #   python src/hpc_mistral_server.py --port 8001

    python src/hpc_mistral_batch_runner.py                 # new task
    python src/hpc_mistral_batch_runner.py 1               # resume task 1
    python src/hpc_mistral_batch_runner.py --server http://localhost:8001
    python src/hpc_mistral_batch_runner.py --song "Bad Ketchup"  # single-song test
    python src/hpc_mistral_batch_runner.py --difficulty Hard
"""

import argparse
import base64
import csv
import datetime
import json
import os
import sys
import time

import librosa
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DIFFICULTY  = "Medium"
MODEL_NAME  = "Mistral-7B"
SERVER_URL  = "http://localhost:8001"

# Songs directory (same Fraxtil collection as the Qwen runner)
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements",
)

# Task registry — separate from Qwen's to avoid collisions
_REGISTRY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".task_registry_mistral.json",
)


# ── Task registry helpers ──────────────────────────────────────────────────────
def load_registry() -> dict:
    if os.path.exists(_REGISTRY_FILE):
        try:
            with open(_REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_task_id": 0, "tasks": {}}


def save_registry(reg: dict) -> None:
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


def create_new_task(reg: dict, difficulty: str) -> int:
    new_id = reg["last_task_id"] + 1
    reg["last_task_id"] = new_id
    reg["tasks"][str(new_id)] = {
        "created": datetime.datetime.now().isoformat(),
        "difficulty": difficulty,
    }
    save_registry(reg)
    return new_id


# ── Audio helpers ──────────────────────────────────────────────────────────────
def audio_to_b64(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── File discovery ─────────────────────────────────────────────────────────────
def get_audio_files(base_dir: str) -> list[str]:
    files = []
    for root, _, fnames in os.walk(base_dir):
        for fname in fnames:
            if fname.lower().endswith((".ogg", ".mp3", ".wav")):
                files.append(os.path.join(root, fname))
    return sorted(files)


# ── Song processing ────────────────────────────────────────────────────────────
def process_song(audio_path: str, task_id: int, server_url: str, difficulty: str) -> None:
    name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    song_dir    = os.path.dirname(audio_path)
    task_tag    = f"task{task_id:04d}"

    print(f"  Processing: {name_no_ext}")

    # Skip if already done for this task
    existing = [
        f for f in os.listdir(song_dir)
        if f.startswith(f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_")
        and f.endswith(".txt")
    ]
    if existing:
        print(f"  Skipping — already done ({task_tag}): {existing[0]}")
        return

    try:
        duration = librosa.get_duration(path=audio_path)
        print(f"  Duration: {duration:.1f}s  | Encoding audio...")

        audio_b64 = audio_to_b64(audio_path)

        print(f"  Sending to Mistral server ({server_url})...")
        resp = requests.post(
            f"{server_url}/generate",
            json={
                "audio_b64":      audio_b64,
                "audio_filename": os.path.basename(audio_path),
                "difficulty":     difficulty,
                "max_new_tokens": 8192,
            },
            timeout=600,   # 10-minute timeout per song
        )
        resp.raise_for_status()
        raw_text = resp.json()["text"]

        if not raw_text.strip():
            print("  ⚠️  Empty response from model.")
            return

        # ── Parse JSON response → list of note rows ────────────────────────
        notes: list[str] = []
        try:
            data = json.loads(raw_text)

            # Unwrap outer object if model returned {"rows": [...]} etc.
            if isinstance(data, dict):
                for key in ("rows", "beatmap", "steps", "data", "chart"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "notes" in item:
                        notes.append(item["notes"])
                    elif isinstance(item, str) and (len(item) == 4 or item == ","):
                        notes.append(item)
            else:
                raise ValueError("Unexpected JSON structure")

        except (json.JSONDecodeError, ValueError):
            # Fallback: treat raw text lines as notes
            notes = [
                line.strip()
                for line in raw_text.splitlines()
                if line.strip() and (len(line.strip()) == 4 or line.strip() == ",")
            ]

        if not notes:
            print("  ⚠️  Could not parse any notes from response.")
            return

        # ── Save files ─────────────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base      = f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_{timestamp}"

        # Plain beatmap .txt
        txt_path = os.path.join(song_dir, f"{base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for note in notes:
                f.write(f"{note}\n")
        print(f"  Beatmap  → {os.path.basename(txt_path)}")

        # Structured .csv (matches Gemini / Qwen output format)
        csv_path = os.path.join(song_dir, f"{base}.csv")
        rows_per_sec = 8.0  # approx. 8 subdivisions/s at 120 BPM
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_ms", "beat_position", "notes",
                "placement_type", "note_type", "confidence", "instrument",
            ])
            beat_idx = 0.0
            for i, note in enumerate(notes):
                if note == ",":
                    beat_idx += 1.0
                    continue
                time_ms  = round((i / rows_per_sec) * 1000, 2)
                beat_pos = round(beat_idx + (i % int(rows_per_sec)) / rows_per_sec, 4)
                writer.writerow([time_ms, beat_pos, note, 1, 3, 0.9, "mixed"])
        print(f"  Full CSV → {os.path.basename(csv_path)}")
        print("  ✅ Done.")

    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot reach server at {server_url}. Is it running?")
    except Exception as exc:
        print(f"  ❌ Error: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mistral batch runner — queries hpc_mistral_server.py"
    )
    parser.add_argument(
        "task_id", nargs="?", type=int,
        help="Resume a specific task ID (omit to create a new task)",
    )
    parser.add_argument("--server", default=SERVER_URL, help="Mistral server URL")
    parser.add_argument("--difficulty", default=DIFFICULTY,
                        choices=["Easy", "Medium", "Hard", "Expert"],
                        help="Beatmap difficulty (default: Medium)")
    parser.add_argument("--song", default=None,
                        help="Test with one song — partial name match, e.g. 'Bad Ketchup'")
    args = parser.parse_args()

    # ── Health check ──────────────────────────────────────────────────────
    try:
        health = requests.get(f"{args.server}/health", timeout=5).json()
        if not health.get("model_loaded"):
            print("⚠️  Server is up but model not loaded yet — waiting for it...")
    except Exception:
        print(f"❌ Cannot reach Mistral server at {args.server}.")
        print("   Start it first:  python src/hpc_mistral_server.py --port 8001")
        sys.exit(1)

    # ── Task management ───────────────────────────────────────────────────
    registry = load_registry()
    if args.task_id:
        task_id = args.task_id
        if str(task_id) not in registry["tasks"]:
            print(f"Error: Task ID {task_id} not in registry.")
            sys.exit(1)
        print(f"▶  Resuming Mistral Task ID: {task_id:04d}")
        difficulty = registry["tasks"][str(task_id)].get("difficulty", args.difficulty)
    else:
        task_id = create_new_task(registry, args.difficulty)
        difficulty = args.difficulty
        print(f"▶  New Mistral Task ID: {task_id:04d}")

    print(f"   Difficulty : {difficulty}")
    print(f"   Server     : {args.server}\n")

    # ── File discovery ────────────────────────────────────────────────────
    target_files = get_audio_files(BASE_DIR)

    if args.song:
        target_files = [
            f for f in target_files
            if args.song.lower() in os.path.basename(os.path.dirname(f)).lower()
               or args.song.lower() in os.path.basename(f).lower()
        ]
        if not target_files:
            print(f"❌ No audio file matching '{args.song}'. Check the song name.")
            sys.exit(1)
        print(f"🎵 Single-song test: {os.path.basename(target_files[0])}\n")
    else:
        print(f"Found {len(target_files)} audio files.\n")

    # ── Process loop ──────────────────────────────────────────────────────
    for i, audio_file in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] ", end="")
        process_song(audio_file, task_id, args.server, difficulty)
        time.sleep(1)


if __name__ == "__main__":
    main()
