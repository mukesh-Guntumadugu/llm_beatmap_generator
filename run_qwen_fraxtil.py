
import os
import sys
import time
import json
import datetime
import librosa
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
from src.beatmap_prompt import build_qwen_prompt

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
        print(f"  Duration: {duration:.1f}s")

        # ── Same prompt as Gemini (imported from src/beatmap_prompt.py) ──────
        prompt_text = build_qwen_prompt(DIFFICULTY, duration)

        print(f"  Sending to Qwen ({MODEL_NAME})...")
        response_text = generate_beatmap_with_qwen(audio_path, prompt=prompt_text)

        if not response_text:
            print("  ❌ No output generated.")
            return

        # Filename: SongName_Difficulty_ModelName_taskXXXX_timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{name_no_ext}_{DIFFICULTY}_{safe_model}_{task_tag}_{timestamp}"

        # ── File 1: raw .txt (full model response) ──────────────────────────
        txt_path = os.path.join(dirname, f"{base}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"  Raw TXT  → {base}.txt")

        # ── File 2: .csv with header matching Gemini format ─────────────────
        csv_path = os.path.join(dirname, f"{base}.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("time_ms,beat_position,notes,placement_type,note_type,confidence,instrument\n")
            for line in response_text.splitlines():
                line = line.strip()
                if line:
                    f.write(line + "\n")
        print(f"  Full CSV → {base}.csv")
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
