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
import time
import csv
import json
import base64
import datetime
import argparse
import requests
import librosa

DIFFICULTY  = "Medium"
MODEL_NAME  = "Qwen2-Audio-7B"
SERVER_URL  = "http://localhost:8000"

# Where the Fraxtil songs live (relative to project root)
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# Task registry (same format as batch_process_fraxtil.py)
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".task_registry_hpc.json")

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(duration: float, difficulty: str = DIFFICULTY) -> str:
    return (
        f"The audio is {duration:.1f} seconds long.\n"
        f"Generate a {difficulty} difficulty StepMania beatmap for the ENTIRE duration.\n\n"
        "Output a JSON array of objects. Each object has:\n"
        "  - notes (str): 4-character row e.g. '1000' OR ',' for measure separator\n\n"
        "Rules:\n"
        "- Each measure has 4, 8, or 16 note rows then a ',' separator.\n"
        "- Use '0000' for silent slots.\n"
        "- Cover the ENTIRE audio. DO NOT STOP EARLY.\n\n"
        "Example:\n"
        '[{"notes":"1000"},{"notes":"0000"},{"notes":"0010"},{"notes":"0000"},{"notes":","}]\n'
    )

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

def create_new_task(reg) -> int:
    new_id = reg["last_task_id"] + 1
    reg["last_task_id"] = new_id
    reg["tasks"][str(new_id)] = {
        "created": datetime.datetime.now().isoformat(),
        "difficulty": DIFFICULTY,
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

def process_song(audio_path: str, task_id: int, server_url: str):
    name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    dirname     = os.path.dirname(audio_path)
    task_tag    = f"task{task_id:04d}"

    print(f"Processing: {name_no_ext}")

    # Skip if already done for this task
    existing = [
        f for f in os.listdir(dirname)
        if f.startswith(f"{name_no_ext}_{DIFFICULTY}_{MODEL_NAME}_{task_tag}_") and f.endswith(".txt")
    ]
    if existing:
        print(f"  Skipping — already done ({task_tag}): {existing[0]}")
        return

    try:
        duration = librosa.get_duration(path=audio_path)
        print(f"  Duration: {duration:.1f}s  |  Encoding audio...")

        audio_b64 = audio_to_b64(audio_path)
        prompt    = build_prompt(duration, DIFFICULTY)

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

        # ── Parse JSON response → list of note rows ───────────────────────
        try:
            data  = json.loads(text)
            notes = [item["notes"] for item in data]
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat raw text lines as notes
            notes = [line.strip() for line in text.splitlines() if line.strip()]

        # ── Save files ────────────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base      = f"{name_no_ext}_{DIFFICULTY}_{MODEL_NAME}_{task_tag}_{timestamp}"

        # Plain beatmap .txt
        txt_path = os.path.join(dirname, f"{base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for note in notes:
                f.write(f"{note}\n")
        print(f"  Beatmap → {os.path.basename(txt_path)}")

        print("  Done.")

    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot reach server at {server_url}. Is it running?")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", nargs="?", type=int, help="Resume a specific task ID")
    parser.add_argument("--server", default=SERVER_URL, help="Qwen server URL")
    args = parser.parse_args()

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
        task_id = create_new_task(registry)
        print(f"▶  New HPC Task ID: {task_id:04d}")

    print(f"   Difficulty : {DIFFICULTY}\n")

    target_files = get_target_files(BASE_DIR)
    print(f"Found {len(target_files)} audio files.\n")

    for i, audio_file in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] ", end="")
        process_song(audio_file, task_id, args.server)
        time.sleep(1)

if __name__ == "__main__":
    main()
