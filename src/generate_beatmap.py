import torch
import torch.nn as nn
import csv
import logging
import sys
import os

# --- CONFIG ---
USE_WANDB = False  # Set to True to enable W&B tracking
ORIGINAL_BEATMAP_PATH = "src/musicForBeatmap/Springtime/beatmap_easy.text"  # For comparison

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- MODEL DEFINITION (Must Match train.py) ---
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

# --- MAPS ---
IDX_TO_CHAR = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
DENSITY_MAP = {0: 4, 1: 8, 2: 12, 3: 16}

def generate_chart(token_path, model_path, output_path):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f"Using Device: {device}")
    
    # 1. Load Tokens
    logging.info(f"Loading Tokens: {token_path}")
    tokens = []
    with open(token_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip Header
        for row in reader:
            tokens.append([int(x) for x in row[1:]])
    
    # Pre-pad 3 seconds (225 frames)
    padding = [[0]*32] * 225
    tokens = padding + tokens
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    num_frames = tokens_tensor.shape[0]
    
    # 2. Load Model
    logging.info(f"Loading Model: {model_path}")
    model = BeatmapLSTM(num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Generate
    window_size = 300
    target_len = 75
    stride = 75
    
    # Calculate number of windows
    # Start at 0 (which maps to padded region). 
    # We want to cover the whole song.
    # tokens len includes padding.
    # Data starts at index 225.
    # Verify: Train loop used 'target_start'.
    # We want to predict starting from frame 0 of audio?
    # Window 0: Input [0:300] (225 pad + 75 audio). Target: Audio [0:75].
    # YES.
    
    generated_lines = []
    
    with torch.no_grad():
        cursor = 0
        while cursor + window_size <= num_frames:
            # Prepare Input
            x = tokens_tensor[cursor : cursor + window_size].unsqueeze(0).to(device) # (1, 300, 32)
            
            # Predict
            notes_out, density_out = model(x)
            
            # Slice Last 16 Frames (Symbolic Slots)
            notes_slice = notes_out[:, -16:, :, :] # (1, 16, 4, 5)
            density_pred = density_out[:, -1, :]   # (1, 4)
            
            # Predictions
            d_idx = torch.argmax(density_pred, dim=1).item()
            num_lines_in_sec = DENSITY_MAP.get(d_idx, 16)
            
            n_preds = torch.argmax(notes_slice, dim=3).cpu().numpy()[0] # (16, 4)
            
            # Extract 'num_lines'
            # These lines represent 1 second (75 frames).
            # We append them to the chart.
            
            for i in range(num_lines_in_sec):
                row_idx = n_preds[i]
                row_str = "".join([IDX_TO_CHAR[c] for c in row_idx])
                generated_lines.append(row_str)
                
            cursor += stride
            
    # 4. Write Output
    logging.info(f"Generated {len(generated_lines)} chart lines.")
    with open(output_path, 'w') as f:
        f.write("\n".join(generated_lines))
    logging.info(f"Written to: {output_path}")
    return generated_lines

def generate_with_wandb(token_path, model_path, output_path, 
                        original_beatmap_path=None,
                        project_name='beatmap-generation',
                        run_name=None):
    """
    Generate beatmap with Weights & Biases tracking and comparison metrics.
    
    Features:
    - W&B dashboard with generation metrics
    - Comparison with original beatmap (if provided)
    - Confusion matrices and accuracy visualizations
    - Per-direction analytics
    - Shareable results via URL
    
    Install: pip install wandb (already installed for training)
    Login: wandb login (already done for training)
    """
    try:
        import wandb
        import datetime
        from generation.beatmap_comparison import (
            load_beatmap, calculate_all_metrics
        )
    except ImportError as e:
        if "wandb" in str(e):
            logging.error("❌ wandb not installed. Run: pip install wandb")
            logging.error("Falling back to regular generation...")
        else:
            logging.error(f"❌ Missing module: {e}")
            logging.error("Make sure beatmap_comparison.py is in src/generation/")
        return generate_chart(token_path, model_path, output_path)
    
    # Extract info for W&B
    song_name = os.path.basename(token_path).split("tokens_")[1].split("_202")[0] if "tokens_" in token_path else "unknown"
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{song_name}_generation_{timestamp}"
    
    # Initialize W&B
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'token_path': token_path,
            'model_path': model_path,
            'song_name': song_name,
            'has_comparison': original_beatmap_path is not None
        }
    )
    
    logging.info(f"{'='*60}")
    logging.info(f"GENERATING WITH WANDB")
    logging.info(f"🔗 Dashboard: {wandb.run.url}")
    logging.info(f"{'='*60}")
    
    # Generate beatmap (reuse existing logic)
    generated_lines = generate_chart(token_path, model_path, output_path)
    
    # Log basic statistics
    total_notes = sum(sum(1 for c in line if c != '0') for line in generated_lines)
    
    wandb.log({
        'total_lines': len(generated_lines),
        'total_notes': total_notes,
        'notes_per_line': total_notes / len(generated_lines) if generated_lines else 0
    })
    
    # Density distribution
    density_counts = {4: 0, 8: 0, 12: 0, 16: 0}
    # Estimate density by counting non-zero lines per second (rough approximation)
    # For visualization purposes
    wandb.log({
        'generation_timestamp': datetime.datetime.now().isoformat(),
        'output_path': output_path
    })
    
    # Compare with original if provided
    if original_beatmap_path and os.path.exists(original_beatmap_path):
        logging.info(f"\nComparing with original: {original_beatmap_path}")
        
        try:
            original_lines = load_beatmap(original_beatmap_path)
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
                '': metrics['density_ratio']
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
            wandb.run.summary['best_accuracy'] = metrics['exact_match_accuracy']
            wandb.run.summary['timing_accuracy'] = metrics['timing_accuracy']
            wandb.run.summary['total_notes_generated'] = metrics['generated_total_notes']
            
            logging.info(f"\n{'='*60}")
            logging.info(f"COMPARISON RESULTS")
            logging.info(f"{'='*60}")
            logging.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
            logging.info(f"Timing Accuracy: {metrics['timing_accuracy']:.2f}%")
            logging.info(f"Density Accuracy: {metrics['density_accuracy']:.2f}%")
            logging.info(f"Notes Generated: {metrics['generated_total_notes']}")
            logging.info(f"Notes Original: {metrics['original_total_notes']}")
            logging.info(f"{'='*60}")
            
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            wandb.log({'comparison_error': str(e)})
    
    # Save generated beatmap as W&B artifact
    artifact = wandb.Artifact(f'{song_name}_beatmap', type='beatmap')
    artifact.add_file(output_path)
    wandb.log_artifact(artifact)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation Complete!")
    logging.info(f"View results: {wandb.run.url}")
    logging.info(f"{'='*60}")
    
    wandb.finish()
    return generated_lines

if __name__ == "__main__":
    t_path = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127032805.csv"
    m_path = "src/training/beatmap_lstm_springtime.pth"
    o_path = "src/generated_notedata.txt"
    
    # Choose generation method
    if USE_WANDB:
        # W&B generation with comparison tracking
        generate_with_wandb(t_path, m_path, o_path, 
                           original_beatmap_path=ORIGINAL_BEATMAP_PATH)
    else:
        # Regular generation (original method)
        generate_chart(t_path, m_path, o_path)

