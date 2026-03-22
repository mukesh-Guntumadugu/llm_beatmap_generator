import os
import glob
import json
import librosa
import soundfile as sf
import numpy as np

SONG_LIST = [
    "Bad Ketchup",
    "Bitch Clap",
    "Call Me Maybe (Remix)",
    "Dead Hollywood",
    "Deal With It",
    "Gangnam Style",
    "Girls",
    "Goin' Under",
    "Goodbye (2012 Mix)",
    "Human Error"
]

BASE_DIR = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"
OUTPUT_DIR = "/Users/mukeshguntumadugu/llm_beatmap_generator/sft_dataset"
AUDIO_OUT_DIR = os.path.join(OUTPUT_DIR, "audio")

CHUNK_DURATION = 20.0  # seconds

def parse_original_onsets(csv_file):
    """Parses the original_onsets_*.csv file to extract timestamps in seconds."""
    onsets_sec = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header: onset_index,onset_ms
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        ms = float(parts[1])
                        onsets_sec.add(round(ms / 1000.0, 3))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error parsing onsets {csv_file}: {e}")
    return sorted(list(onsets_sec))

def main():
    jsonl_data = []
    
    for song in SONG_LIST:
        print(f"Processing {song}...")
        song_dir = os.path.join(BASE_DIR, song)
        if not os.path.exists(song_dir):
            print(f"  Missing directory for {song}")
            continue
            
        # Find audio
        audio_files = glob.glob(os.path.join(song_dir, "*.ogg")) + glob.glob(os.path.join(song_dir, "*.mp3"))
        if not audio_files:
            continue
        audio_path = audio_files[0]
        
        # Find librosa original onsets CSV
        onset_files = glob.glob(os.path.join(song_dir, "original_onsets_*.csv"))
        if not onset_files:
            print(f"  Missing original_onsets_*.csv for {song}")
            continue
        onset_path = onset_files[0]
        
        onsets = parse_original_onsets(onset_path)
        if not onsets:
            print(f"  No onsets found for {song}")
            continue
            
        print(f"  Loaded {len(onsets)} onsets for {song}")
        
        # Load audio and chunk
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        for chunk_idx, start_time in enumerate(np.arange(0, duration, CHUNK_DURATION)):
            end_time = min(start_time + CHUNK_DURATION, duration)
            if end_time - start_time < 5.0: # Skip very short final chunks
                continue
                
            # Filter onsets for this chunk
            chunk_onsets = [round(o - start_time, 3) for o in onsets if start_time <= o < end_time]
            if len(chunk_onsets) == 0:
                continue # Skip empty chunks
                
            # Save audio chunk
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            y_chunk = y[start_frame:end_frame]
            
            chunk_filename = f"{song.replace(' ', '_')}_{chunk_idx:03d}.wav"
            chunk_out_path = os.path.join(AUDIO_OUT_DIR, chunk_filename)
            sf.write(chunk_out_path, y_chunk, sr)
            
            # Create Qwen2-Audio JSONL entry
            # Qwen2-Audio expects conversational format
            onset_str = ", ".join(map(str, chunk_onsets))
            entry = {
                "id": f"{song.replace(' ', '_')}_{chunk_idx:03d}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": os.path.abspath(chunk_out_path)},
                            {"type": "text", "text": "List the onsets in this audio segment as a comma-separated list of timestamps in seconds."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": onset_str}
                        ]
                    }
                ]
            }
            jsonl_data.append(entry)
            
    # Save JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nCreated {len(jsonl_data)} audio-text pairs for SFT.")
    print(f"Dataset saved to {jsonl_path}")

if __name__ == "__main__":
    main()
