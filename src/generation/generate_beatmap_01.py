import torch
import torch.nn as nn
import csv
import numpy as np
import logging
import sys
import os
import datetime
import re

# Optional: librosa for audio BPM detection
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not installed. Audio BPM detection disabled. Install with: pip install librosa")

# --- SONG-SPECIFIC CONFIG (Change these for each new song) ---
SSC_FILE = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ssc"  # StepMania chart file
AUDIO_FILE = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg"  # Original audio (mp3/ogg/wav)
ORIGINAL_BEATMAP_PATH = "src/musicForBeatmap/MechaTribe Assault/MechaTribe_eassy.txt"  # Ground truth for comparison

# --- GENERAL CONFIG ---
USE_WANDB = True  # Set to True to enable W&B tracking

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- BPM AUTO-DETECTION HELPERS ---
def get_bpm_from_ssc(ssc_path: str) -> float:
    """
    Extract BPM from StepMania .ssc file.
    
    Args:
        ssc_path: Path to .ssc file
        
    Returns:
        BPM as float
        
    Raises:
        FileNotFoundError: If .ssc file doesn't exist
        ValueError: If BPM not found in file
    """
    if not os.path.exists(ssc_path):
        raise FileNotFoundError(f"SSC file not found: {ssc_path}")
    
    with open(ssc_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('#BPMS:'):
                # Format: #BPMS:0.000=180.000; or #BPMS:0.000000=180.000000
                match = re.search(r'=(\d+\.\d+)', line)
                if match:
                    bpm = float(match.group(1))
                    logging.info(f"✓ BPM from .ssc: {bpm}")
                    return bpm
    
    raise ValueError(f"BPM not found in {ssc_path}")

def detect_bpm_from_audio(audio_path: str) -> float:
    """
    Detect BPM from audio file using librosa.
    
    Args:
        audio_path: Path to audio file (mp3, ogg, wav, etc.)
        
    Returns:
        Detected BPM as float
        
    Raises:
        ImportError: If librosa is not installed
        FileNotFoundError: If audio file doesn't exist
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required for audio BPM detection. Install with: pip install librosa")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    logging.info(f"Detecting BPM from audio: {audio_path}...")
    y, sr = librosa.load(audio_path, duration=60)  # Analyze first 60 seconds
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    logging.info(f"✓ BPM from audio: {bpm:.2f}")
    return bpm

def auto_detect_bpm(ssc_path: str = None, audio_path: str = None, tolerance: float = 2.0) -> float:
    """
    Automatically detect BPM with cross-verification.
    
    Priority:
    1. If .ssc file provided, use that (most accurate)
    2. If audio file provided, detect from audio
    3. Cross-verify both if available and warn on mismatch
    
    Args:
        ssc_path: Path to .ssc file (optional)
        audio_path: Path to audio file (optional)
        tolerance: Maximum allowed difference for cross-verification (BPM)
        
    Returns:
        Detected BPM as float
        
    Raises:
        ValueError: If no valid source provided or BPM detection fails
    """
    bpm_ssc = None
    bpm_audio = None
    
    # Try .ssc first (most reliable)
    if ssc_path:
        try:
            bpm_ssc = get_bpm_from_ssc(ssc_path)
        except Exception as e:
            logging.warning(f"Could not extract BPM from .ssc: {e}")
    
    # Try audio detection
    if audio_path and LIBROSA_AVAILABLE:
        try:
            bpm_audio = detect_bpm_from_audio(audio_path)
        except Exception as e:
            logging.warning(f"Could not detect BPM from audio: {e}")
    
    # Cross-verify if both available
    if bpm_ssc is not None and bpm_audio is not None:
        diff = abs(bpm_ssc - bpm_audio)
        if diff > tolerance:
            logging.warning(f"⚠️  BPM MISMATCH: .ssc={bpm_ssc}, audio={bpm_audio:.2f} (diff={diff:.2f})")
            logging.warning(f"Using .ssc value ({bpm_ssc}) as it's more reliable")
        else:
            logging.info(f"✓ BPM verified: .ssc={bpm_ssc}, audio={bpm_audio:.2f} (match within {tolerance} BPM)")
        return bpm_ssc
    
    # Use whichever is available
    if bpm_ssc is not None:
        return bpm_ssc
    if bpm_audio is not None:
        return bpm_audio
    
    raise ValueError("Could not detect BPM from any source. Please provide .ssc or audio file.")

# --- MODEL DEFINITION (Must match train.py) ---
class BeatmapLSTM(nn.Module):
    def __init__(self, codebook_size=1024, embed_dim=128, hidden_dim=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.dropout_layer = nn.Dropout(0.5)
        self.fc_notes = nn.Linear(hidden_dim, 16 * 4 * 5) 
        self.fc_density = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = self.embedding(x) 
        out, (h_n, c_n) = self.lstm(x) 
        final_out = out[:, -1, :] 
        final_out = self.dropout_layer(final_out)
        notes_out = self.fc_notes(final_out) 
        notes_out = notes_out.view(-1, 16, 4, 5) 
        density_out = self.fc_density(final_out) 
        return notes_out, density_out

def generate():
    # --- CONFIG ---
    TOKEN_FILE = "src/Neural Audio Codecs/outputs/Mecha-Tribe Assault_20260211_005257_tokens.csv"
    # right now it is trained on only this generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv
    # we generated beatmap we got 80% accuracy. 

    # src/Neural Audio Codecs/outputs/Mecha-Tribe Assault_20260211_005257_tokens.csv
    # TOKEN_FILE = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv"
    MODEL_PATH = "src/training/beatmap_lstm_springtime_best.pth"
    
    # "src/training/beatmap_lstm_springtime_best.pth" right now it was not trained in this  Mecha-Tribe Assault_ song.

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract song name from ORIGINAL_BEATMAP_PATH
    song_name = os.path.splitext(os.path.basename(ORIGINAL_BEATMAP_PATH))[0]
    OUTPUT_FILE = f"generated_{song_name}_{timestamp}.txt"
    
    FRAMES_PER_SEC = 75
    WINDOW_FRAMES = 300
    STRIDE_FRAMES = 75 # Predict 1 second at a time
    CODEBOOKS = 4
    
    # Auto-detect BPM from .ssc and audio files
    logging.info("="*60)
    logging.info("DETECTING BPM")
    logging.info("="*60)
    BPM = auto_detect_bpm(ssc_path=SSC_FILE, audio_path=AUDIO_FILE)
    logging.info(f"✓ Using BPM: {BPM}")
    logging.info("="*60)
    
    FRAMES_PER_MEASURE = (4 * 60 / BPM) * FRAMES_PER_SEC
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # 1. LOAD MODEL
    model = BeatmapLSTM()
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logging.info("Model loaded.")
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")
        return

    # 2. LOAD TOKENS
    logging.info(f"Loading tokens from {TOKEN_FILE}...")
    tokens_list = []
    with open(TOKEN_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Header
        for row in reader:
            vals = [int(x) for x in row[1:1+CODEBOOKS]]
            tokens_list.append(vals)
    
    raw_tokens = torch.tensor(tokens_list, dtype=torch.long) # (TotalFrames, 4)
    num_frames = raw_tokens.shape[0]
    logging.info(f"Total Frames: {num_frames}")

    # 3. PREDICT TIMELINE
    # Timeline: stores predicted note (0-4) for each frame. 
    # Initialize with 0 (No note)
    timeline_notes = torch.zeros((num_frames, 4), dtype=torch.long)
    density_timeline = torch.zeros((num_frames), dtype=torch.long)  # Store density predictions
    
    logging.info("Generating predictions...")
    
    # Iterate with sliding window (EXACTLY like train.py's SpringtimeDataset)
    # Window: 300 frames input → predict 1 second (75 frames) output
    # Stride: 75 frames
    
    for i in range(0, num_frames, STRIDE_FRAMES):
        input_start = i
        input_end = i + WINDOW_FRAMES
        target_start = i
        target_end = min(i + STRIDE_FRAMES, num_frames)
        
        # Prepare Input (EXACTLY like SpringtimeDataset.__getitem__)
        if input_end > num_frames:
            # Pad if necessary
            valid_len = num_frames - input_start
            chunk = raw_tokens[input_start:num_frames]
            padding = torch.zeros((WINDOW_FRAMES - valid_len, CODEBOOKS), dtype=torch.long)
            chunk = torch.cat([chunk, padding], dim=0)  # Shape: (300, 4)
        else:
            chunk = raw_tokens[input_start:input_end]  # Shape: (300, 4)
        
        # Flatten to 1D (EXACTLY like train.py line 158)
        x_flat = chunk.view(-1)  # Shape: (1200,) — NOT (1, 1200)!
        
        # Add batch dimension (like DataLoader does)
        x_batch = x_flat.unsqueeze(0).to(device)  # Shape: (1, 1200)
        
        with torch.no_grad():
            notes_out, density_out = model(x_batch)
            
        # Decode Output
        pred_start = 0
        pred_len = target_end - target_start
        
        # Determine Density
        d_idx = torch.argmax(density_out, dim=1).item()
        d_map = {0: 4, 1: 8, 2: 12, 3: 16}
        num_lines = d_map.get(d_idx, 16)
        
        # Store density for this segment
        for frame_idx in range(target_start, target_end):
            if frame_idx < num_frames:
                density_timeline[frame_idx] = d_idx
        
        # Extract Lines
        notes_batch = torch.argmax(notes_out, dim=3)[0] # (16, 4)
        active_lines = notes_batch[:num_lines]
        
        # Map Lines to Frames
        # We have 'num_lines' to distribute over 'STRIDE_FRAMES' (75)
        # Simple nearest neighbor mapping
        step = STRIDE_FRAMES / num_lines
        
        for ln_idx in range(num_lines):
            frame_offset = int(ln_idx * step)
            if frame_offset < pred_len:
                abs_frame = target_start + frame_offset
                if abs_frame < num_frames:
                    timeline_notes[abs_frame] = active_lines[ln_idx].cpu()
                    
        if i % (STRIDE_FRAMES * 10) == 0:
            print(f"Processed {i}/{num_frames} frames...", end='\r')
            
    logging.info("\nTimeline prediction complete.")
    
    # 4. RECONSTRUCT MEASURES
    logging.info(f"Reconstructing Measures (BPM {BPM}, Frames/Measure {FRAMES_PER_MEASURE:.2f})...")
    
    output_lines = []
    audio_ptr = 0.0
    
    measure_count = 0
    while audio_ptr < num_frames:
        start_f = int(audio_ptr)
        end_f = int(audio_ptr + FRAMES_PER_MEASURE)
        end_f = min(end_f, num_frames)
        
        if end_f <= start_f: break
        
        # Extract segment
        segment = timeline_notes[start_f:end_f]
        
        # Determine density for this measure from predictions
        # Use the most common density in this measure
        measure_densities = density_timeline[start_f:end_f]
        if len(measure_densities) > 0:
            # Get most common density class in this measure
            density_class = torch.mode(measure_densities).values.item()
            d_map = {0: 4, 1: 8, 2: 12, 3: 16}
            OUTPUT_RES = d_map.get(density_class, 16)
        else:
            OUTPUT_RES = 16  # Default fallback
        
        slots = []
        step_f = FRAMES_PER_MEASURE / OUTPUT_RES
        
        for k in range(OUTPUT_RES):
            # Check frame window for ANY note
            # Window: [start_f + k*step, start_f + (k+1)*step]
            w_start = int(start_f + k*step_f)
            w_end = int(start_f + (k+1)*step_f)
            
            # Find dominant note in this window
            # Priority: Look for non-zeros.
            chunk = timeline_notes[w_start:w_end]
            
            # If any note exists, pick it.
            # Collision handling: Priority?
            # Simple max.
            found = False
            for r in chunk:
                if torch.sum(r) > 0: # Non-zero row
                    slots.append("".join(str(x.item()) for x in r))
                    found = True
                    break
            if not found:
                slots.append("0000")
                
        # Write measure
        for l in slots:
            output_lines.append(l)
        output_lines.append(",") # End Measure
        
        measure_count += 1
        audio_ptr += FRAMES_PER_MEASURE

    # 5. SAVE
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(output_lines))
        
    logging.info(f"Saved. this is beatmap :  ----> : {measure_count} measures to {OUTPUT_FILE}")
    return OUTPUT_FILE, output_lines

def generate_with_wandb(project_name='beatmap-generation', run_name=None):
    """
    Generate beatmap with Weights & Biases tracking and comparison metrics.
   
    Features:
    - W&B dashboard with generation metrics
    - Comparison with original beatmap
    - Confusion matrices and accuracy visualizations
    - Per-direction analytics
    - Shareable results via URL
    """
    try:
        import wandb
        from beatmap_comparison import (
            load_beatmap, 
            calculate_all_metrics,
            calculate_temporal_metrics,
            plot_accuracy_over_time,
            generate_density_confusion_matrix,
            generate_concurrency_confusion_matrix
        )
    except ImportError as e:
        if "wandb" in str(e):
            logging.error("❌ wandb not installed. Run: pip install wandb")
            logging.error("Falling back to regular generation...")
        else:
            logging.error(f"❌ Missing module: {e}")
            logging.error("Make sure beatmap_comparison.py is in src/generation/")
        return generate()
    
    # Initialize W&B
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    song_name = os.path.splitext(os.path.basename(ORIGINAL_BEATMAP_PATH))[0]
    if run_name is None:
        run_name = f"{song_name}_generation_{timestamp}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'song': song_name,
            'model': 'beatmap_lstm_best',
            'has_comparison': USE_WANDB and os.path.exists(ORIGINAL_BEATMAP_PATH)
        }
    )
    
    logging.info(f"{'='*60}")
    logging.info(f"GENERATING WITH WANDB")
    logging.info(f"🔗 Dashboard: {wandb.run.url}")
    logging.info(f"{'='*60}")
    
    # Generate beatmap (reuse existing logic)
    output_file, generated_lines = generate()
    
    # Log basic statistics
    total_notes = sum(sum(1 for c in line if c != '0' and c != ',') for line in generated_lines)
    total_lines = len([l for l in generated_lines if l != ','])
    
    wandb.log({
        'total_lines': total_lines,
        'total_notes': total_notes,
        'notes_per_line': total_notes / total_lines if total_lines else 0,
        'output_file': output_file
    })
    
    # Compare with original if configured
    if ORIGINAL_BEATMAP_PATH and os.path.exists(ORIGINAL_BEATMAP_PATH):
        logging.info(f"\nComparing with original: {ORIGINAL_BEATMAP_PATH}")
        
        try:
            original_lines = load_beatmap(ORIGINAL_BEATMAP_PATH)
            metrics = calculate_all_metrics(generated_lines, original_lines)
            
            # Log accuracy metrics
            wandb.log({
                'accuracy/exact_match': metrics['exact_match_accuracy'],
                'accuracy/timing_alignment': metrics['timing_accuracy'],
                'accuracy/density': metrics['density_accuracy'],
                'accuracy/left': metrics['left_accuracy'],
                'accuracy/down': metrics['down_accuracy'],
                'accuracy/up': metrics['up_accuracy'],
                'accuracy/right': metrics['right_accuracy'],
                'notes/generated': metrics['generated_total_notes'],
                'notes/original': metrics['original_total_notes'],
                'notes/density_ratio': metrics['density_ratio']
            })
            
            # Log per-direction metrics
            for direction, scores in metrics['per_direction'].items():
                wandb.log({
                    f'direction/{direction}/precision': scores['precision'],
                    f'direction/{direction}/recall': scores['recall'],
                    f'direction/{direction}/f1_score': scores['f1_score']
                })
            
            # Log confusion matrix
            cm_data = metrics['confusion_matrix']
            wandb.log({
                'confusion_matrix/notes': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=cm_data['y_true'],
                    preds=cm_data['y_pred'],
                    class_names=cm_data['labels']
                )
            })
            
            # Summary statistics
            wandb.run.summary['exact_match_accuracy'] = metrics['exact_match_accuracy']
            wandb.run.summary['timing_accuracy'] = metrics['timing_accuracy']
            wandb.run.summary['density_accuracy'] = metrics['density_accuracy']
            
            logging.info(f"\n{'='*60}")
            logging.info(f"COMPARISON RESULTS")
            logging.info(f"{'='*60}")
            logging.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
            logging.info(f"Timing Accuracy: {metrics['timing_accuracy']:.2f}%")
            logging.info(f"Density Accuracy: {metrics['density_accuracy']:.2f}%")
            logging.info(f"{'='*60}")
            
            # Calculate and log temporal (measure-by-measure) metrics
            logging.info(f"\n{'='*60}")
            logging.info(f"TEMPORAL ANALYSIS (Measure-by-Measure)")
            logging.info(f"{'='*60}")
            
            temporal_metrics = calculate_temporal_metrics(output_file, ORIGINAL_BEATMAP_PATH)
            
            # Log temporal metrics to W&B
            wandb.log({
                'temporal/overall_accuracy': temporal_metrics['temporal_accuracy'],
                'temporal/std_deviation': temporal_metrics['temporal_std'],
                'temporal/total_measures': temporal_metrics['total_measures'],
                'temporal/perfect_measures': temporal_metrics['perfect_measures'],
                'temporal/perfect_ratio': temporal_metrics['perfect_measure_ratio']
            })
            
            # Log best and worst sections
            wandb.run.summary['best_measure'] = temporal_metrics['best_measure']['measure_num']
            wandb.run.summary['best_measure_accuracy'] = temporal_metrics['best_measure']['accuracy']
            wandb.run.summary['worst_measure'] = temporal_metrics['worst_measure']['measure_num']
            wandb.run.summary['worst_measure_accuracy'] = temporal_metrics['worst_measure']['accuracy']
            
            # Create and upload accuracy over time visualization
            # Match naming convention of training loss graphs
            plot_path = f'src/generation/accuracy_over_time_{song_name}_{timestamp}.png'
            plot_accuracy_over_time(
                temporal_metrics['measure_accuracies'],
                measure_details=temporal_metrics['measure_details'],  # Pass detailed metrics
                output_path=plot_path,
                title='Beatmap Generation: Comprehensive Metrics Over Time'
            )
            
            # Log visualization to W&B
            wandb.log({'temporal/accuracy_over_time': wandb.Image(plot_path)})
            
            # Generate and log density confusion matrix (EXACT line counts: 4, 8, 12, 16, 20, 24)
            density_cm = generate_density_confusion_matrix(output_file, ORIGINAL_BEATMAP_PATH)
            wandb.log({
                'confusion_matrix/density': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=density_cm['y_true'],
                    preds=density_cm['y_pred'],
                    class_names=density_cm['labels']
                )
            })
            
            # Generate and log note concurrency confusion matrix
            concurrency_cm = generate_concurrency_confusion_matrix(generated_lines, original_lines)
            wandb.log({
                'confusion_matrix/concurrency': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=concurrency_cm['y_true'],
                    preds=concurrency_cm['y_pred'],
                    class_names=concurrency_cm['labels']
                )
            })
            
            # Print summary
            logging.info(f"Overall Temporal Accuracy: {temporal_metrics['temporal_accuracy']:.2f}%")
            logging.info(f"Accuracy Std Dev: {temporal_metrics['temporal_std']:.2f}%")
            logging.info(f"Total Measures: {temporal_metrics['total_measures']}")
            logging.info(f"Perfect Measures: {temporal_metrics['perfect_measures']} ({temporal_metrics['perfect_measure_ratio']:.1f}%)")
            logging.info(f"Best Section: Measure {temporal_metrics['best_measure']['measure_num']} ({temporal_metrics['best_measure']['accuracy']:.1f}%)")
            logging.info(f"Worst Section: Measure {temporal_metrics['worst_measure']['measure_num']} ({temporal_metrics['worst_measure']['accuracy']:.1f}%)")
            logging.info(f"{'='*60}")
            
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            wandb.log({'comparison_error': str(e)})
    
    # Save generated beatmap as W&B artifact
    artifact = wandb.Artifact(f'{song_name}_beatmap', type='beatmap')
    artifact.add_file(output_file)
    wandb.log_artifact(artifact)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation Complete!")
    logging.info(f"View results: {wandb.run.url}")
    logging.info(f"{'='*60}")
    
    wandb.finish()
    return output_file, generated_lines

if __name__ == "__main__":
    if USE_WANDB:
        # W&B generation with comparison tracking
        generate_with_wandb()
    else:
        # Regular generation (original method)
        generate()
