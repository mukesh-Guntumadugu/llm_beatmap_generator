import re
from pathlib import Path

def parse_ssc_file(ssc_path, target_difficulty="Easy", target_stepstype="dance-single"):
    """
    Parses SSC file to extract notes for a specific chart.
    Returns a list of (Time, Frame, NoteString).
    """
    with open(ssc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract Global Tags
    offset_match = re.search(r"#OFFSET:([-0-9.]+);", content)
    bpms_match = re.search(r"#BPMS:(.+?);", content)
    
    if not offset_match or not bpms_match:
        raise ValueError("Could not find OFFSET or BPMS")
        
    offset = float(offset_match.group(1))
    
    # We assume constant BPM for now based on user file (0.000=180.000)
    # A full parser would handle changes, but let's stick to the prompt's file.
    bpms_str = bpms_match.group(1)
    # simple parse: "0.000=180.000"
    first_bpm = float(bpms_str.split('=')[1])
    seconds_per_beat = 60.0 / first_bpm
    
    print(f"Global Offset: {offset}")
    print(f"BPM: {first_bpm}")
    
    # 2. Find the specific Chart
    # We split by #NOTEDATA:; to separate charts
    charts = content.split("#NOTEDATA:;")
    
    target_chart = None
    for chart in charts:
        if f"#STEPSTYPE:{target_stepstype};" in chart and f"#DIFFICULTY:{target_difficulty};" in chart:
            target_chart = chart
            break
            
    if not target_chart:
        raise ValueError(f"Chart {target_stepstype} / {target_difficulty} not found.")
        
    print(f"Found Chart: {target_stepstype} - {target_difficulty}")
    
    # 3. Extract Notes
    # Notes are between #NOTES: and ;
    notes_match = re.search(r"#NOTES:(.*?);", target_chart, re.DOTALL)
    if not notes_match:
        raise ValueError("Could not find #NOTES section")
        
    notes_raw = notes_match.group(1).strip()
    
    # 4. Parse Measures
    measures = notes_raw.split(',')
    
    parsed_notes = []
    
    # Audio Frame Rate
    FRAME_RATE = 75.0
    
    current_beat = 0.0
    
    for measure_idx, measure in enumerate(measures):
        rows = measure.strip().split('\n')
        # Remove comments or empty lines if any (though usually clean)
        rows = [r.strip() for r in rows if len(r.strip()) >= 4 and not r.strip().startswith('//')]
        
        num_rows = len(rows)
        if num_rows == 0:
            continue
            
        # Each measure is 4 beats (standard 4/4)
        beats_per_measure = 4.0
        
        for i, row in enumerate(rows):
            # Calculate Beat
            # Row index i out of num_rows covers the 4 beats
            beat_within_measure = (i / num_rows) * beats_per_measure
            total_beat = (measure_idx * beats_per_measure) + beat_within_measure
            
            # Calculate Time
            # Time = Offset + (TotalBeat * SecondsPerBeat)
            time_sec = offset + (total_beat * seconds_per_beat)
            
            # Align to Frame
            frame_idx = int(round(time_sec * FRAME_RATE))
            
            # If note is not empty "0000"
            if row != "0000":
                parsed_notes.append({
                    "time": time_sec,
                    "frame": frame_idx,
                    "measure": measure_idx,
                    "beat": total_beat,
                    "note": row
                })
                
    return parsed_notes

def create_aligned_dataset(ssc_path, tokens_path, output_path):
    import torch
    
    # 1. Load Tokens
    print(f"Loading tokens from {tokens_path}...")
    tokens = torch.load(tokens_path)
    # Shape: (1, 32, T) or (32, T)
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0) # (32, T)
        
    num_frames = tokens.shape[1]
    print(f"Audio Tokens: {num_frames} frames")
    
    # 2. Parse Notes
    print(f"Parsing SSC: {ssc_path}...")
    notes = parse_ssc_file(ssc_path)
    print(f"Parsed {len(notes)} notes.")
    
    # 3. Create Target Tensor
    # Shape: (T, 4) - 4 lanes
    # Values: 0=Empty, 1=Tap, 2=Hold, 3=Tail, 4=Roll, 5=Mine
    # Mapping Chars to Ints
    # M -> 5, F -> 0 (ignore), L -> 1 (treat as tap for now?)
    char_map = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'M': 5
    }
    
    targets = torch.zeros((num_frames, 4), dtype=torch.long)
    
    for n in notes:
        frame = n['frame']
        note_str = n['note'] # e.g. "1000"
        
        if 0 <= frame < num_frames:
            for lane, char in enumerate(note_str):
                val = char_map.get(char, 0) # Default to 0 if unknown
                targets[frame, lane] = val
        else:
            # Note is outside audio duration (rare but possible with padding)
            pass
            
    # 4. Save Dataset
    dataset = {
        "tokens": tokens,   # (32, T)
        "targets": targets, # (T, 4)
        "metadata": {
            "ssc_path": str(ssc_path),
            "tokens_path": str(tokens_path)
        }
    }
    
    torch.save(dataset, output_path)
    print(f"Dataset saved to {output_path}")
    print(f"Target Tensor Shape: {targets.shape}")
    print(f"Non-zero frames in target: {(targets.sum(dim=1) > 0).sum()}")

if __name__ == "__main__":
    ssc_path = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ssc"
    # Update this to the actual token file path found dynamically or hardcoded for now
    tokens_path = "src/Neural Audio Codecs/outputs/Mecha-Tribe Assault_20260121_010831_tokens.pt"
    output_dataset = "src/Neural Audio Codecs/outputs/Mecha-Tribe Assault_dataset.pt"
    
    try:
        create_aligned_dataset(ssc_path, tokens_path, output_dataset)
    except Exception as e:
        print(f"Error: {e}")
