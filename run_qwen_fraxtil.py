
import os
import time
import datetime
import librosa
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

DIFFICULTY = "Easy"
MODEL_NAME = "Qwen2-Audio-7B-Instruct"

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "-")
    
    new_filename = f"{name_no_ext}_{name_no_ext}_{timestamp}_{difficulty}_{safe_model_name}.txt"
    return os.path.join(dirname, new_filename)

def process_song(audio_path):
    print(f"Processing: {os.path.basename(audio_path)}")
    
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

        print(f"  Sending to Qwen ({MODEL_NAME})...")
        
        # Call Qwen Interface
        response_text = generate_beatmap_with_qwen(audio_path, prompt=prompt_text)
        
        if not response_text:
            print("  ❌ No output generated.")
            return

        # Save to File
        output_path = generate_filename(audio_path, DIFFICULTY, MODEL_NAME)
        print(f"  Saving to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
                
        print("  ✅ Done.")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    # Setup Qwen (loads model once)
    try:
        setup_qwen()
    except Exception as e:
        print(f"Critial Error loading model: {e}")
        return

    base_dir = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
    print(f"Scanning directory: {base_dir}")
    
    target_files = get_target_files(base_dir)
    print(f"Found {len(target_files)} audio files.")
    
    for i, audio_file in enumerate(target_files):
        print(f"\n[{i+1}/{len(target_files)}] Processing file...")
        process_song(audio_file)
        time.sleep(1)

if __name__ == "__main__":
    main()
