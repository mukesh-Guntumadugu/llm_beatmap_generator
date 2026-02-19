
import os
import time
import datetime
import json
from dotenv import load_dotenv
import librosa

# Import GenAI functions from existing module
# Make sure src is in python path or run from root
# Import GenAI functions from existing module
# Make sure src is in python path or run from root
from src.gemini import setup_gemini, generate_beatmap_chunk, Beat

DIFFICULTY = "Easy"
MODEL_NAME = "gemini-pro-latest" # Options: gemini-pro-latest, gemini-2.0-flash-001

def get_target_files(base_dir):
    """Recursively find all .ogg, .mp3, .wav files."""
    audio_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    return audio_files

def generate_filename(audio_path, difficulty, model_name):
    """
    Generates filename format: {OriginalName}_{OriginalName}_{Timestamp}_{Difficulty}_{Model}.txt
    """
    dirname = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    name_no_ext = os.path.splitext(basename)[0]
    
    # Timestamp format: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct new filename
    # Example: Bad Ketchup_Bad Ketchup_20240218_123000_Easy_gemini-pro-latest.txt
    new_filename = f"{name_no_ext}_{name_no_ext}_{timestamp}_{difficulty}_{model_name}.txt"
    return os.path.join(dirname, new_filename)

def process_song(audio_path):
    print(f"Processing: {os.path.basename(audio_path)}")
    
    try:
        # 1. Get Duration
        duration = librosa.get_duration(path=audio_path)
        expected_commas = int(duration)
        print(f"  Duration: {duration:.1f}s")
        
        # 2. Construct Prompt (Identical to gemini.py 'Easy' prompt)
        prompt_text = (
            f"The audio is {duration:.1f} seconds long. You MUST generate chart data for the ENTIRE duration.\n"
            f"Target: approximately {expected_commas} measure separators (commas) — one for roughly every second of audio.\n\n"
            f"Listen to the audio and generate StepMania chart rows for a {DIFFICULTY} difficulty. "
            "Output a continuous sequence of 4-character strings covering the entire audio duration. "
            "Each string represents a row in the chart (Left, Down, Up, Right). "
            "Use '0000' for empty rows to maintain correct timing and rhythm (e.g., 4 rows per beat). "
            "IMPORTANT: Separate measures with a comma ',' on its own line/entry. A measure usually has 4/4 beats , 8 .\n"
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
            ""
            
        )

        # 3. Generate
        print(f"  Sending to Gemini ({DIFFICULTY}, {MODEL_NAME})...")
        beats = generate_beatmap_chunk(audio_path, prompt=prompt_text, model_name=MODEL_NAME)
        
        if not beats:
            print("  ❌ No beats generated.")
            return

        # 4. Save to File
        output_path = generate_filename(audio_path, DIFFICULTY, MODEL_NAME)
        print(f"  Saving to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for beat in beats:
                f.write(f"{beat.notes}\n")
                
        print("  ✅ Done.")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    load_dotenv(override=True)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return
    
    setup_gemini(api_key)
    
    base_dir = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
    print(f"Scanning directory: {base_dir}")
    
    target_files = get_target_files(base_dir)
    print(f"Found {len(target_files)} audio files.")
    
    for i, audio_file in enumerate(target_files):
        print(f"\n[{i+1}/{len(target_files)}] Processing file...")
        process_song(audio_file)
        
        # Rate limiting / Courtesy pause
        time.sleep(2)

if __name__ == "__main__":
    main()
