
import os
import sys
import time
import json
import datetime
import librosa
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
import numpy as np

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
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Duration: {duration:.1f}s")

        # ── Calculate BPM and Onsets ──
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        expected_commas = int(duration)
        onset_times_str = ", ".join([f"{t:.2f}" for t in onset_times])

        # ── Inline Prompt ──
        prompt_text = (
            f"The audio is {duration:.1f} seconds long. You MUST generate chart data for the ENTIRE duration.\n"
            f"Target: approximately {expected_commas} measure separators (commas) — one for roughly every second of audio.\n\n"
            f"The detected BPM is approximately {bpm:.1f}.\n"
            f"Here are the detected onset times (in seconds) where significant musical events happen:\n[{onset_times_str}]\n\n"
            "Listen to the audio and generate StepMania chart rows for a {DIFFICULTY} difficulty. "
            "Output a continuous sequence of 4-character strings covering the entire audio duration. "
            "Each string represents a row in the chart (Left, Down, Up, Right). "
            "Use '0000' for empty rows to maintain correct timing and rhythm (e.g., 4 rows per beat). "
            "IMPORTANT: Separate measures with a comma ',' on its own line/entry. A measure usually has 4 / 8 / 12 beats.\n"
            "Use the following note codes:\n"
            "0: Empty\n"
            "1: Tap\n"
            "2: Hold Head\n"
            "3: Hold End\n"
            "4: Roll Head\n"
            ",: Measure Separator\n\n"
            "=== STEP PLACEMENT LOGIC ===\n"
            "When placing notes, ensure they align logically with the music rhythm using these concepts:\n"
            "- ONSET: A note placed exactly on a major audio event / transient. Often aligns with strong notes.\n"
            "- BEAT: A note placed exactly on a primary BPM beat (quarter note downbeat).\n"
            "- GRID: A note placed on a standard subdivision (eighth, sixteenth) between downbeats.\n"
            "- PERCUSSIVE: A note placed on a sharp drum hit (kick, snare, hi-hat).\n"
            "- UNALIGNED: Avoid unaligned notes that do not match the beat grid or an audio onset.\n\n"
            "=== COMMON DDR PATTERNS FOR REFERENCE ===\n"
            "Match the intensity of the music using these patterns:\n"
            "- JUMP: 2 arrows hit simultaneously (e.g. '1001' or '0110').\n"
            "- JACK: The same single arrow hit rapidly twice in a row (e.g. '1000' then '1000').\n"
            "- DOUBLE STEP: The same arrow hit twice, but separated by other notes (e.g. '1000', '0001', '1000').\n"
            "- STREAM: Dense alternating singles using all 4 columns, no jacks (e.g. '1000', '0100', '0010', '0001').\n"
            "- FOOTSWITCH: Strict alternation between Left-side panels (L/D) and Right-side panels (U/R).\n"
            "- STAIRCASE/SPIN: A circular rotational flow (e.g. '1000', '0100', '0001', '0010').\n"
            "- BRACKET: One foot hitting two adjacent panels simultaneously (L+D '1100', or U+R '0011').\n"
            "- CANDLE: A triplet rhythm of Single->Jump->Single (e.g. '1000', '1001', '0100').\n"
            "- GALLOP: A quick alternating burst sandwiching jumps and singles.\n\n"
            "Example Sequence:\n"
            "1000\n"
            "0020\n"
            "0000\n"
            "0030\n"
            ",\n"
            "0001\n"
            "1001\n"
            "0000\n"
            "0020\n"
            "1000\n"
            "0030\n"
            "1001\n"
            "0110\n"
            ",\n"
            "Focus on strong downbeats using mostly taps. Ensure commas separate logical musical chunks.\n"
            "DO NOT STOP EARLY. GENERATE UNTIL THE END OF THE SONG."
        )

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

        # ── File 2: Clean CSV (notes only) ──
        csv_path = os.path.join(dirname, f"{base}.csv")
        valid_rows = []
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("notes\n")
            for line in response_text.splitlines():
                line = line.strip()
                if line == "," or (len(line) == 4 and all(c in '01234' for c in line)):
                    f.write(line + "\n")
                    valid_rows.append(line)
        print(f"  Notes CSV → {base}.csv")

        # ── File 3: Structured JSON (Converted post-generation) ──
        json_path = os.path.join(dirname, f"{base}.json")
        structured_json = []
        current_beat = 1.0 # Beat 1.0 represents 0.0 time_ms
        
        # Group notes into measures
        measures = []
        current_measure = []
        for line in valid_rows:
            if line == ",":
                if current_measure:
                    measures.append(current_measure)
                    current_measure = []
                measures.append(",")
            else:
                current_measure.append(line)
        if current_measure:
            measures.append(current_measure)

        for measure in measures:
            if measure == ",":
                time_ms = ((current_beat - 1.0) / bpm) * 60 * 1000
                structured_json.append({
                    "time_ms": round(time_ms, 2),
                    "beat_position": round(current_beat, 3),
                    "notes": ",",
                    "placement_type": -1,
                    "note_type": -1,
                    "confidence": 1.0,
                    "instrument": "separator"
                })
            else:
                n_rows = len(measure)
                if n_rows == 0: continue
                beats_per_row = 4.0 / n_rows # Each measure = 4 beats
                for row in measure:
                    time_ms = ((current_beat - 1.0) / bpm) * 60 * 1000
                    structured_json.append({
                        "time_ms": round(time_ms, 2),
                        "beat_position": round(current_beat, 3),
                        "notes": row,
                        "placement_type": 0, # Unsure
                        "note_type": 3, # Default
                        "confidence": 1.0,
                        "instrument": "unknown"
                    })
                    current_beat += beats_per_row
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_json, f, indent=2)
        print(f"  Final JSON → {base}.json")
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
