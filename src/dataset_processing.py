import torch
import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine
from simfile.notes import NoteData
from pathlib import Path
import argparse
import numpy as np

class BeatmapTokenizer:
    """
    Class to process .ssc files and align them with audio tokens.
    """
    def __init__(self, frame_rate=75.0):
        """
        Args:
            frame_rate (float): The frame rate of the EnCodec model (tokens per second).
                                EnCodec 24khz uses 75Hz by default.
        """
        self.frame_rate = frame_rate

    def parse_ssc(self, ssc_path):
        """
        Parses an .ssc file and extracts note data.
        Returns the first available chart (usually the highest difficulty if ordered).
        """
        with open(ssc_path, 'r', encoding='utf-8') as f:
            ssc = simfile.load(f)
        
        # Taking the first chart for now. In future we might want to select specific difficulty.
        if not ssc.charts:
            raise ValueError(f"No charts found in {ssc_path}")
            
        chart = ssc.charts[0]
        print(f"Processing Chart: {chart.difficulty} ({chart.meter})")
        
        return ssc, chart

    def get_note_timings(self, ssc, chart):
        """
        Extracts the time (in seconds) and column index for every note.
        Returns a list of tuples: (time, column_index, note_type)
        """
        note_data = NoteData(chart)
        # TimingEngine handles the complex beat-to-time logic
        timing_engine = TimingEngine(TimingData(ssc, chart))
        
        notes = []
        
        # Iterate over notes
        # simfile's NoteData iterator yields (beat, column, note_type)
        for note in note_data:
            beat = note.beat
            col = note.column
            note_type = note.note_type
            
            # 1 = Tap, 2 = Hold Head, 4 = Roll Head. We ignore mines (M) usually.
            if note_type.name in ['TAP', 'HOLD_HEAD', 'ROLL_HEAD']:
                time = timing_engine.time_at(beat)
                notes.append((time, col, 1)) # 1 indicates "Hit"
        
        return notes

    def align_to_audio(self, notes, num_audio_tokens):
        """
        Creates a label tensor aligned to the audio tokens.
        
        Args:
            notes (list): List of (time, col, type).
            num_audio_tokens (int): Total number of time steps in the audio tokens.
            
        Returns:
            torch.Tensor: Shape (num_audio_tokens, 4). 4 columns for Left, Down, Up, Right.
                          Values are 0 or 1.
        """
        # 4 Lanes for standard DDR/StepMania
        labels = torch.zeros((num_audio_tokens, 4), dtype=torch.float32)
        
        ignored_count = 0
        
        for time, col, val in notes:
            # Convert time (seconds) to token index
            # Index = Time * FrameRate
            token_idx = int(round(time * self.frame_rate))
            
            if 0 <= token_idx < num_audio_tokens:
                if 0 <= col < 4:
                    labels[token_idx, col] = 1.0
            else:
                ignored_count += 1
                
        if ignored_count > 0:
            print(f"Warning: {ignored_count} notes were out of bounds of the audio duration.")
            
        return labels

def process_song(token_path, ssc_path, output_path):
    print(f"--- Processing {Path(token_path).stem} ---")
    
    # 1. Load Audio Tokens
    tokens = torch.load(token_path)
    # Tokens shape: (1, 8, Time)
    num_tokens = tokens.shape[-1]
    duration_est = num_tokens / 75.0
    print(f"Audio Duration: {duration_est:.2f}s ({num_tokens} tokens)")
    
    # 2. Parse Beatmap
    processor = BeatmapTokenizer(frame_rate=75.0)
    ssc, chart = processor.parse_ssc(ssc_path)
    notes = processor.get_note_timings(ssc, chart)
    print(f"Found {len(notes)} notes.")
    
    # 3. Create Labels
    labels = processor.align_to_audio(notes, num_tokens)
    print(f"Label Tensor Shape: {labels.shape}")
    
    # 4. Save Combined Dataset
    dataset = {
        'audio_tokens': tokens, # (1, 8, T)
        'beatmap_labels': labels, # (T, 4)
        'metadata': {
            'song_title': ssc.title,
            'artist': ssc.artist,
            'difficulty': chart.difficulty,
            'frame_rate': 75.0
        }
    }
    
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    
    # Validation Print
    # Find a spot with notes
    active_indices = torch.where(labels.sum(dim=1) > 0)[0]
    if len(active_indices) > 0:
        idx = active_indices[0]
        print(f"\nVerification Example at Token {idx} ({idx/75.0:.2f}s):")
        print(f"Audio Code (Book 0): {tokens[0, 0, idx]}")
        print(f"Label (L,D,U,R): {labels[idx].tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("token_path", help="Path to .pt audio tokens")
    parser.add_argument("ssc_path", help="Path to .ssc file")
    parser.add_argument("--output", help="Output path for dataset .pt", default=None)
    
    args = parser.parse_args()
    
    out_path = args.output
    if out_path is None:
        # Default: outputs/Song_dataset.pt
        p = Path(args.token_path)
        out_path = p.parent / f"{p.stem.replace('_tokens', '')}_dataset.pt"
        
    process_song(args.token_path, args.ssc_path, out_path)
