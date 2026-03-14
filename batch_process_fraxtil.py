
import os
import sys
import time
import csv
import json
import datetime
from dotenv import load_dotenv
import librosa

from src.gemini import setup_gemini, generate_beatmap_csv_chunked, create_beatmap_prompt_cache

MODEL_NAME    = "gemini-3-flash-preview"
CHUNK_DURATION = 30.0  # seconds per chunk — Flash has smaller context, so we chunk and combine

# ── Task ID registry ──────────────────────────────────────────────────────────
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".task_registry.json")

def load_registry() -> dict:
    if os.path.exists(_REGISTRY_FILE):
        try:
            with open(_REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"last_task_id": 0, "tasks": {}}

def save_registry(registry: dict):
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def create_new_task(registry: dict, difficulty: str) -> int:
    """Increments the task counter and registers a new task. Returns the new task ID."""
    new_id = registry["last_task_id"] + 1
    registry["last_task_id"] = new_id
    registry["tasks"][str(new_id)] = {
        "created": datetime.datetime.now().isoformat(),
        "difficulty": difficulty,
        "model": MODEL_NAME,
    }
    save_registry(registry)
    return new_id


def get_target_files(base_dir):
    """Recursively find all .ogg, .mp3, .wav files."""
    audio_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                audio_files.append(os.path.join(root, file))
    return audio_files


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


def process_song(audio_path, task_id: int, difficulty: str, cached_content_name=None):
    print(f"Processing: {os.path.basename(audio_path)}")

    dirname = os.path.dirname(audio_path)
    name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    task_tag = f"task{task_id:04d}"  # e.g. task0003

    # Skip if this song was already done under the same task ID
    existing = [
        f for f in os.listdir(dirname)
        if f.startswith(f"{name_no_ext}_{difficulty}_{task_tag}_") and f.endswith(".txt")
    ]
    if existing:
        print(f"  Skipping — already generated (task {task_id}): {existing[0]}")
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
            print(f"  Duration: {duration:.1f}s  |  BPM: {bpm:.1f}")
        else:
            print(f"  Duration: {duration:.1f}s")

        cache_status = "cached prompt" if cached_content_name else "full prompt"
        print(f"  Sending chunked to Gemini ({difficulty}, {MODEL_NAME}, {cache_status})...")
        rows = generate_beatmap_csv_chunked(
            audio_path=audio_path,
            duration=duration,
            difficulty=difficulty,
            model_name=MODEL_NAME,
            chunk_duration=CHUNK_DURATION,
            cached_content_name=cached_content_name,
            bpm=bpm
        )

        if not rows:
            print("   No data generated.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Filename format: SongName_Beginner_task0003_20260224_012345
        base = f"{name_no_ext}_{difficulty}_{MODEL_NAME}_{task_tag}_{timestamp}"

        # ── File 1: plain beatmap .txt ─────────────────────────────────────
        beatmap_path = os.path.join(dirname, f"{base}.txt")
        with open(beatmap_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f"{row.notes}\n")
        print(f"  Beatmap  → {os.path.basename(beatmap_path)}")

        # ── File 2: rich metadata .csv (same base name) ────────────────────
        csv_path = os.path.join(dirname, f"{base}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_ms", "beat_position", "notes",
                "placement_type", "note_type", "confidence", "instrument"
            ])
            for row in rows:
                writer.writerow([
                    row.time_ms,
                    f"{row.beat_position:.3f}",
                    row.notes,
                    row.placement_type,
                    row.note_type,
                    f"{row.confidence:.3f}",
                    row.instrument
                ])
        print(f"  Full CSV → {os.path.basename(csv_path)}")
        print("   Done.")

    except Exception as e:
        print(f"   Error: {e}")


def main():
    load_dotenv(override=True)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    setup_gemini(api_key)

    # ── Task ID: resume existing or create new ────────────────────────────
    registry = load_registry()

    parser = __import__('argparse').ArgumentParser()
    parser.add_argument("task_id", nargs="?", type=int)
    parser.add_argument("--song", default=None, help="Test with one song (partial name, e.g. 'Bad Ketchup')")
    parser.add_argument("--difficulty", default="Hard", choices=["Beginner", "Easy", "Medium", "Hard", "Challenge"], help="Difficulty level to generate")
    args = parser.parse_args()
    difficulty = args.difficulty

    if args.task_id:
        task_id = args.task_id
        if str(task_id) not in registry["tasks"]:
            print(f"Error: Task ID {task_id} not found in registry.")
            return
        print(f"▶  Resuming Task ID: {task_id:04d}  (skipping already-processed songs)")
    else:
        task_id = create_new_task(registry, difficulty)
        print(f"▶  New Task ID: {task_id:04d}  (saved to .task_registry.json)")

    print(f"   Difficulty : {difficulty}")
    print(f"   Model      : {MODEL_NAME}\n")

    # ── Create prompt cache ONCE for the entire batch run ─────────────────
    cache_name = create_beatmap_prompt_cache(
        difficulty=difficulty,
        model_name=MODEL_NAME,
        ttl_seconds=3600
    )
    if cache_name:
        print(f"Using context cache: {cache_name}\n")
    else:
        print("Running without cache (full prompt per song).\n")

    base_dir = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
    print(f"Scanning directory: {base_dir}")

    target_files = get_target_files(base_dir)

    # Single-song test mode
    if args.song:
        target_files = [f for f in target_files if args.song.lower() in os.path.basename(os.path.dirname(f)).lower()]
        if not target_files:
            print(f"\u274c No audio file found matching '{args.song}'.")
            return
        print(f"\U0001f3b5 Single-song test: {os.path.basename(target_files[0])}\n")
    else:
        print(f"Found {len(target_files)} audio files.\n")

    for i, audio_file in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] ", end="")
        process_song(audio_file, task_id=task_id, difficulty=difficulty, cached_content_name=cache_name)
        time.sleep(2)  # Rate limiting pause

if __name__ == "__main__":
    main()
