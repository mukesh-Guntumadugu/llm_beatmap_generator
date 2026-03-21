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

def parse_sm_metadata(sm_file):
    metadata = {'bpm': None, 'offset': 0.0}
    try:
        with open(sm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("#OFFSET:"):
                    try:
                        val = line.split(":")[1].split(";")[0].strip()
                        metadata['offset'] = float(val)
                    except ValueError:
                        pass
                if line.strip().startswith("#BPMS:"):
                    try:
                        val = line.split(":")[1].split(";")[0].strip()
                        first_bpm = val.split(",")[0]
                        bpm_val = first_bpm.split("=")[1]
                        metadata['bpm'] = float(bpm_val)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"Error parsing metadata {sm_file}: {e}")
    return metadata

def parse_ssc_notes(file_path, target_difficulty="Hard"):
    """Parses .ssc and returns measures for a target difficulty. Fallbacks to Challenge/Medium."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    charts = {}
    in_chart = False
    in_notes = False
    temp_difficulty = "Unknown"
    current_measures = []
    current_measure = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("#NOTEDATA:"):
            in_chart = True
            in_notes = False
            temp_difficulty = "Unknown"
            current_measures = []
            current_measure = []
            continue
            
        if in_chart:
            if line.startswith("#DIFFICULTY:"):
                temp_difficulty = line.replace("#DIFFICULTY:", "").replace(";", "").strip()
                
            if line.startswith("#NOTES:"):
                in_notes = True
                continue
                
            if in_notes:
                has_semicolon = ";" in line
                note_line = line.replace(";", "").split("//")[0].strip()
                
                if note_line == ",":
                    current_measures.append(current_measure)
                    current_measure = []
                elif len(note_line) >= 4 and all(c in '01234M' for c in note_line[:4]):
                    current_measure.append(note_line[:4])
                
                if has_semicolon or line.startswith(";"):
                    if current_measure:
                        current_measures.append(current_measure)
                    charts[temp_difficulty] = current_measures
                    in_notes = False
                    in_chart = False
                    
    # Pick difficulty
    if target_difficulty in charts:
        return charts[target_difficulty]
    elif "Challenge" in charts:
        return charts["Challenge"]
    elif "Medium" in charts:
        return charts["Medium"]
    elif charts:
        return list(charts.values())[-1]
    
    return []

def get_note_times(measures, bpm, offset):
    note_times = set()
    beats_per_measure = 4
    if bpm is None or bpm <= 0: return []
    seconds_per_beat = 60.0 / bpm
    start_time = offset 
    
    for measure_idx, measure in enumerate(measures):
        measure_start_beat = measure_idx * beats_per_measure
        lines_in_measure = len(measure)
        if lines_in_measure == 0: continue
        
        beats_per_line = beats_per_measure / lines_in_measure
        
        for line_idx, line in enumerate(measure):
            has_note = any(c in '1234' for c in line) # Ignoring M for pure onset
            if has_note:
                beat_time = measure_start_beat + (line_idx * beats_per_line)
                timestamp = start_time + (beat_time * seconds_per_beat)
                if timestamp >= 0:
                    note_times.add(round(timestamp, 3))
                
    return sorted(list(note_times))

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
        
        # Find SSC
        ssc_files = glob.glob(os.path.join(song_dir, "*.ssc")) + glob.glob(os.path.join(song_dir, "*.sm"))
        if not ssc_files:
            continue
        ssc_path = ssc_files[0]
        
        metadata = parse_sm_metadata(ssc_path)
        measures = parse_ssc_notes(ssc_path)
        
        onsets = get_note_times(measures, metadata['bpm'], metadata['offset'])
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
