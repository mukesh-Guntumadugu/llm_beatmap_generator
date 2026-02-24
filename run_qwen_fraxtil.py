
import os
import sys
import time
import json
import datetime
import librosa
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

DIFFICULTY = "Medium"
MODEL_NAME = "Qwen2-Audio-7B-Instruct"

# ── Task ID registry ──────────────────────────────────────────────────────────
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".task_registry_qwen.json")

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
        "model": MODEL_NAME,
    }
    save_registry(reg)
    return new_id

# ── File helpers ──────────────────────────────────────────────────────────────
def get_target_files(base_dir):
    """Recursively find all .ogg, .mp3, .wav files."""
    audio_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)

# ── Song processing ───────────────────────────────────────────────────────────
def process_song(audio_path, task_id: int):
    name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    dirname     = os.path.dirname(audio_path)
    task_tag    = f"task{task_id:04d}"
    safe_model  = MODEL_NAME.replace("/", "-")

    print(f"Processing: {name_no_ext}")

    # Skip if already done for this task
    existing = [
        f for f in os.listdir(dirname)
        if f.startswith(f"{name_no_ext}_{DIFFICULTY}_{safe_model}_{task_tag}_") and f.endswith(".txt")
    ]
    if existing:
        print(f"  Skipping — already done ({task_tag}): {existing[0]}")
        return

    try:
        duration = librosa.get_duration(path=audio_path)
        expected_commas = int(duration)
        print(f"  Duration: {duration:.1f}s")

        prompt_text = (
            f"The audio is {duration:.1f} seconds long. You MUST generate chart data for the ENTIRE duration.\n"
            f"Target: approximately {expected_commas} measure separators (commas) — one for roughly every second of audio.\n\n"
            f"Listen to the audio and generate StepMania chart rows for a {DIFFICULTY} difficulty. "
            "Output a continuous sequence of 4-character strings covering the entire audio duration. "
            "Each string represents a row in the chart (Left, Down, Up, Right). "
            "Use '0000' for empty rows to maintain correct timing and rhythm (e.g., 4 rows per beat). "
            "IMPORTANT: Separate measures with a comma ',' on its own line/entry. A measure usually has 4/4 beats, 8.\n"
            "Use the following note codes:\n"
            " in one second you can have 4 lines are 8 lines are 16 lines are 12, 32"
            "0: Empty\n"
            "1: Tap\n"
            "2: Hold Head\n"
            "3: Hold End\n"
            "4: Roll Head\n"
            ",: Measure Separator\n\n"
            "Example Sequence:\n"
            "1000\n"
            "4020\n"
            "1001\n"
            "0130\n"
            ",\n"
            "0010\n"
            "...\n"
        )

        print(f"  Sending to Qwen ({MODEL_NAME})...")
        response_text = generate_beatmap_with_qwen(audio_path, prompt=prompt_text)

        if not response_text:
            print("  ❌ No output generated.")
            return

        # Filename: SongName_Difficulty_ModelName_taskXXXX_timestamp.txt
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename    = f"{name_no_ext}_{DIFFICULTY}_{safe_model}_{task_tag}_{timestamp}.txt"
        output_path = os.path.join(dirname, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)

        print(f"  Beatmap → {filename}")
        print("  ✅ Done.")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Task ID: resume existing or create new
    registry = load_registry()
    if len(sys.argv) > 1:
        try:
            task_id = int(sys.argv[1])
            if str(task_id) not in registry["tasks"]:
                print(f"Error: Task ID {task_id} not found in registry.")
                return
            print(f"▶  Resuming Qwen Task ID: {task_id:04d}")
        except ValueError:
            print(f"Error: Invalid task ID '{sys.argv[1]}'.")
            return
    else:
        task_id = create_new_task(registry)
        print(f"▶  New Qwen Task ID: {task_id:04d}")

    print(f"   Difficulty : {DIFFICULTY}")
    print(f"   Model      : {MODEL_NAME}\n")

    # Setup Qwen (loads model once)
    try:
        setup_qwen()
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    base_dir = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
    print(f"Scanning directory: {base_dir}")

    target_files = get_target_files(base_dir)
    print(f"Found {len(target_files)} audio files.\n")

    for i, audio_file in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] ", end="")
        process_song(audio_file, task_id=task_id)
        time.sleep(1)

if __name__ == "__main__":
    main()
