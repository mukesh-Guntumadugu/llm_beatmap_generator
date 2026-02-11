import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import datetime
import os

# Add src to path to import tokenizer if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))

import math

# ... (Previous imports exist)

# Helper for quantization
def quantize_measure(measure_notes, beats_per_measure=4):
    """
    Refines a list of notes in a measure to the lowest valid quantization.
    measure_notes: list of (beat_in_measure, note_string)
    """
    # Grid resolution: 192 ticks per measure (Standard High Res)
    TICKS_PER_MEASURE = 192
    TICKS_PER_BEAT = TICKS_PER_MEASURE / beats_per_measure # 48
    
    # Candidate grids: (Lines per beat, Ticks per line)
    # 4th (1 line/beat, 48 ticks)
    # 8th (2 lines/beat, 24 ticks)
    # 12th (3 lines/beat, 16 ticks) (Triplets)
    # 16th (4 lines/beat, 12 ticks)
    # 24th (6 lines/beat, 8 ticks)
    # 32nd (8 lines/beat, 6 ticks)
    # 48th (12 lines/beat, 4 ticks)
    # 64th (16 lines/beat, 3 ticks)
    # 192nd (48 lines/beat, 1 tick)
    
    candidates = [
        (1, 48), (2, 24), (3, 16), (4, 12), 
        (6, 8), (8, 6), (12, 4), (16, 3), (48, 1)
    ]
    
    best_lines_per_beat = 48 # Fallback
    
    # Pre-convert beats to abstract "ticks" (float)
    note_ticks = []
    for beat, note in measure_notes:
        t = beat * TICKS_PER_BEAT
        note_ticks.append((t, note))
        
    # Find best fit
    for lines_per_beat, tick_step in candidates:
        max_error = 0
        
        for t, note in note_ticks:
            # Distance to nearest grid line
            # nearest_grid_idx = round(t / tick_step)
            # grid_pos = nearest_grid_idx * tick_step
            # dist = abs(t - grid_pos)
            
            dist = abs(t - round(t / tick_step) * tick_step)
            if dist > max_error:
                max_error = dist
                
        # Tolerance: How much jitter do we allow?
        # 75Hz frame is ~0.013s. 
        # At 180 BPM, 1 beat = 0.333s => 48 ticks.
        # 1 tick = 0.007s.
        # So 1 frame is roughly 2 ticks.
        # We should allow ~3-4 ticks of error to be safe?
        # Let's say 2.5 ticks tolerance (approx 1 frame jitter)
        
        if max_error <= 3.0: 
            best_lines_per_beat = lines_per_beat
            break
            
    # Generate Output
    lines_per_measure = best_lines_per_beat * beats_per_measure
    grid = ["0000"] * lines_per_measure
    
    tick_step = TICKS_PER_MEASURE / lines_per_measure
    
    for t, note in note_ticks:
        # Snap to grid
        grid_idx = int(round(t / tick_step))
        if 0 <= grid_idx < lines_per_measure:
            # Handle collision? (e.g. 2 notes snap to same line)
            # Prioritize non-zeros? Logic: OR them? Or overwrite?
            # Overwrite is fine for now, usually sequential.
            # Ideally merge: "1000" + "0010" -> "1010"
            existing = grid[grid_idx]
            if existing != "0000":
                # Merge
                merged = ""
                for i in range(4):
                    c1 = existing[i]
                    c2 = note[i]
                    # If either is not 0, take the non-zero one
                    if c1 != '0': merged += c1
                    elif c2 != '0': merged += c2
                    else: merged += '0'
                grid[grid_idx] = merged
            else:
                grid[grid_idx] = note
                
    return grid

# --- Model Definition (Duplicated for standalone usage) ---
class BeatmapLSTM(nn.Module):
    def __init__(self, num_codebooks=32, codebook_size=1024, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(num_codebooks)
        ])
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc_lanes = nn.ModuleList([
            nn.Linear(hidden_dim, 6) for _ in range(4)
        ])
        
    def forward(self, x):
        # x: (Batch, SeqLen, 32)
        emb_sum = 0
        for i, embedding in enumerate(self.embeddings):
            emb_sum = emb_sum + embedding(x[:, :, i])
        x = self.input_proj(emb_sum)
        out, _ = self.lstm(x)
        logits = []
        for fc in self.fc_lanes:
            logits.append(fc(out))
        return torch.stack(logits, dim=2)

def generate_chart(audio_path, model_path, output_path, timestamp=None, bpm=180.0, offset=-0.012):
    print(f"Generating chart for: {audio_path}")
    print(f"Params: BPM={bpm}, Offset={offset}")
    
    # 1. Tokenize Audio
    # We'll call the EnCodec script or import it. 
    # Importing module with spaces in path using importlib
    import importlib.util
    
    # Define path to the module
    module_path = Path("src/Neural Audio Codecs/EnCodecimplementation.py").resolve()
    spec = importlib.util.spec_from_file_location("EnCodecimplementation", module_path)
    audio_codec_module = importlib.util.module_from_spec(spec)
    sys.modules["EnCodecimplementation"] = audio_codec_module
    spec.loader.exec_module(audio_codec_module)
    
    AudioTokenizer = audio_codec_module.AudioTokenizer
    
    tokenizer = AudioTokenizer(device='cpu', target_bandwidth=24.0)
    tokens = tokenizer.tokenize(audio_path) # (1, 32, Frames)
    
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0) # (32, Frames)
    
    # Transpose for LSTM: (Frames, 32)
    tokens = tokens.t()
    
    # Add batch dim: (1, Frames, 32)
    input_tensor = tokens.unsqueeze(0)
    
    # --- SAVE TOKENS ---
    # Create directory if it doesn't exist
    tokens_dir = Path("src/tokens_generated")
    tokens_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
    song_name = Path(audio_path).stem
    script_name = Path(__file__).stem # get 'generate_chart'
    
    # Format: {script_name}_tokens_{song_name}_{timestamp}
    token_filename_pt = f"{script_name}_Encodex_tokens_{song_name}_{timestamp}.pt"
    token_filename_csv = f"{script_name}_Encodex_tokens_{song_name}_{timestamp}.csv"
    
    token_save_path_pt = tokens_dir / token_filename_pt
    token_save_path_csv = tokens_dir / token_filename_csv
    
    # Save PT
    torch.save(tokens, token_save_path_pt)
    
    # Save CSV Preview
    import csv
    with open(token_save_path_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: Frame, CB_0, CB_1...
        header = ["Frame"] + [f"Codebook_{i}" for i in range(32)]
        writer.writerow(header)
        
        for i, frame in enumerate(tokens):
            row = [i] + frame.tolist()
            writer.writerow(row)
            
    print(f"Tokens tensor saved to {token_save_path_pt}")
    print(f"Tokens CSV saved to {token_save_path_csv}")
    
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = BeatmapLSTM()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(input_tensor) # (1, Frames, 4, 6)
        
    # Get predictions: (1, Frames, 4)
    probs = torch.softmax(logits, dim=-1)
    # Argmax for discrete notes
    preds = torch.argmax(probs, dim=-1).squeeze(0) # (Frames, 4)
    
    # --- MIDDLE WAY PROCESS LOGGING ---
    # User requested: "middle way process" folder and CSV saying "I sent this things to LSTM I got this one as response"
    middle_dir = Path("src/middle_way_process")
    middle_dir.mkdir(parents=True, exist_ok=True)
    
    csv_log_path = middle_dir / f"inputs_outputs_log_{song_name}_{timestamp}.csv"
    
    print(f"Logging inputs and outputs to {csv_log_path}...")
    import csv
    
    with open(csv_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        # Frame, Input_0..31, Output_0..3
        header = ["Frame"] + [f"Input_CB_{i}" for i in range(32)] + ["Output_Lane_0", "Output_Lane_1", "Output_Lane_2", "Output_Lane_3"]
        writer.writerow(header)
        
        # Write rows
        # input_tensor: (1, Frames, 32) -> tokens
        # preds: (Frames, 4)
        
        for i in range(len(tokens)):
            frame_inputs = tokens[i].tolist() # List of 32 ints
            frame_outputs = preds[i].tolist() # List of 4 ints
            
            row = [i] + frame_inputs + frame_outputs
            writer.writerow(row)
            
    print(f"Middle process log saved to {csv_log_path}")
    
    # 4. Smart Formatting
    # Map ints back to chars
    # 0=0, 1=1, 2=2, 3=3, 4=4, 5=M
    int_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '0'} # User requested NO MINES. 5 -> 0
    
    lines = []
    num_frames = preds.shape[0]
    
    FRAME_RATE = 75.0
    seconds_per_beat = 60.0 / bpm
    
    # Group notes by measure
    # A measure is 4 beats
    
    measures_data = {} # measure_idx -> list of (beat_in_measure, note_str)
    
    last_measure_idx = -1
    
    for i in range(num_frames):
        # Decode note
        row_str = ""
        has_note = False
        for lane in range(4):
            val = preds[i, lane].item()
            if val != 0: has_note = True
            row_str += int_map[val]
            
        if not has_note:
            continue
            
        # Convert Frame -> Time -> Beat
        time_sec = i / FRAME_RATE
        # Time = Offset + (TotalBeat * SPB)
        # TotalBeat = (Time - Offset) / SPB
        total_beat = (time_sec - offset) / seconds_per_beat
        
        if total_beat < 0: continue # Skip negative beats (intro padding)
        
        measure_idx = int(total_beat // 4)
        beat_in_measure = total_beat % 4
        
        if measure_idx not in measures_data:
            measures_data[measure_idx] = []
        
        measures_data[measure_idx].append((beat_in_measure, row_str))
        
        if measure_idx > last_measure_idx:
            last_measure_idx = measure_idx
            
    # Write Measures
    output_lines = []
    
    # Handle gaps? StepMania expects measures to be contiguous.
    # If we have measure 0 and measure 5, we must print 1, 2, 3, 4 as empty measures.
    
    for m in range(last_measure_idx + 1):
        notes = measures_data.get(m, [])
        
        if not notes:
            # Empty measure: 4th notes 0000
            output_lines.extend(["0000"] * 4) # Default 4 lines
        else:
            # Quantize
            grid = quantize_measure(notes)
            output_lines.extend(grid)
            
        output_lines.append(",") # End of measure comma
        
    # Replace last comma with semicolon?
    # Usually SM files need comma between measures.
    # The file content usually goes directly into #NOTES: ... ;
    # So removing the very last comma and making it semicolon is safer if we control the full block.
    # The prompt asked for semicolon at end.
    
    if output_lines and output_lines[-1] == ",":
         output_lines.pop()
         
    output_lines.append(";")
            
    # Save
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines))
        
    print(f"Chart saved to {output_path}")
    print(f"Total Lines: {len(output_lines)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--model", default="src/training/beatmap_lstm.pth", help="Path to trained model")
    parser.add_argument("--output", default=None, help="Output file path (optional)")
    parser.add_argument("--bpm", default=180.0, type=float, help="BPM of the song")
    parser.add_argument("--offset", default=-0.012, type=float, help="Global Offset in seconds")
    
    args = parser.parse_args()
    
    output_path = args.output
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    args = parser.parse_args()
    
    # Force Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # ENFORCE OUTPUT STRUCTURE
    # Ignore args.output to prevent user error (as requested to fix the "wrong file" issue)
    beatmap_dir = Path("src/beatmap_generated")
    beatmap_dir.mkdir(parents=True, exist_ok=True)
    
    song_name = Path(args.audio_path).stem
    
    # Filename format: beatmap_generated_chart_{Song}_{Timestamp}.txt
    output_filename = f"beatmap_generated_chart_{song_name}_{timestamp}.txt"
    output_path = beatmap_dir / output_filename
    
    if args.output:
        print(f"NOTE: Ignoring user output argument '{args.output}' to enforce standard naming convention.")
        
    generate_chart(args.audio_path, args.model, output_path, timestamp=timestamp, bpm=args.bpm, offset=args.offset)
