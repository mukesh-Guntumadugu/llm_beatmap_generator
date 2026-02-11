import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import re
import math
import os
import sys
import logging
import datetime
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# --- CONFIG ---
DEBUG_MODE = False  # Set to True for full input/beatmap dumping. False for training summary.
EPOCHS = 600  # Global epochs setting - change here to apply everywhere

# --- Logging Setup ---
def setup_logging(log_dir="src/training/logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_run_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Writing to: {log_file}")
    return log_file

class SpringtimeDataset(Dataset):
    def __init__(self, token_csv_path, ssc_path):
        self.frames_per_sec = 75
        self.window_frames = 300 # INPUT: 4 Seconds of audio context
        self.stride_frames = 75  # STEP: Move 1 second forward each time (Prediction Calculation)
        self.codebooks_to_use = 4 # EnCodec uses 4 parallel codebooks to represent audio. We use all 4.
        
        # 1. Load Tokens from CSV (Audio)
        tokens_list = []
        files_to_load = []

        if os.path.isdir(token_csv_path):
            # Load all CSVs in directory matching pattern
            listing = os.listdir(token_csv_path)
            for fname in listing:
                if fname.startswith("generate_chart_Encodex_tokens_") and fname.endswith(".csv"):
                    files_to_load.append(os.path.join(token_csv_path, fname))
            files_to_load.sort() # Ensure deterministic order
        else:
            files_to_load.append(token_csv_path)

        logging.info(f"Loading {len(files_to_load)} Token Files from: {token_csv_path}")
        
        for fpath in files_to_load:
            try:
                with open(fpath, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader) 
                    for row in reader:
                        # Safety check for empty rows
                        if not row: continue
                        vals = [int(x) for x in row[1:1+self.codebooks_to_use]] 
                        tokens_list.append(vals)
            except Exception as e:
                logging.warning(f"Failed to load {fpath}: {e}")

        raw_tokens = torch.tensor(tokens_list, dtype=torch.long)
        self.num_frames = raw_tokens.shape[0]
        self.tokens_tensor = raw_tokens
        logging.info(f"Loaded {self.num_frames} frames (Total).")

        # 2. Parse Beatmap Text
        logging.info(f"Loading Beatmap Text: {ssc_path}")
        self.labels_tensor = torch.zeros((self.num_frames, 4), dtype=torch.long)
        self.density_tensor = torch.zeros((self.num_frames), dtype=torch.long) 

        try:
            with open(ssc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            measures = content.split(',')
        except Exception as e:
            raise ValueError(f"Failed to read beatmap text: {e}")

        audio_ptr = 0.0
        bpm = 181.685
        seconds_per_beat = 60.0 / bpm
        seconds_per_measure = 4 * seconds_per_beat
        frames_per_measure = seconds_per_measure * self.frames_per_sec # ~99.07 frames
        
        char_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'M': 0}
        
        for measure in measures:
            lines = measure.strip().split('\n')
            # Handle potential trailing semi-colons or whitespace
            lines = [l.strip().replace(';', '') for l in lines if len(l.strip()) >= 4 and not l.strip().startswith('//')]
            
            num_lines = len(lines)
            if num_lines == 0:
                audio_ptr += frames_per_measure
                continue
            
            if num_lines <= 4: d_class = 0
            elif num_lines <= 8: d_class = 1
            elif num_lines <= 12: d_class = 2
            else: d_class = 3
            
            frames_per_line = frames_per_measure / num_lines
            for i, note_line in enumerate(lines):
                start_f = int(audio_ptr + (i * frames_per_line))
                end_f = int(audio_ptr + ((i + 1) * frames_per_line))
                if start_f >= self.num_frames: break
                end_f = min(end_f, self.num_frames)
                
                row_labels = []
                for c in note_line:
                    val = char_map.get(c, 0)
                    row_labels.append(val)
                if len(row_labels) > 4: row_labels = row_labels[:4]
 
                if end_f > start_f:
                    self.labels_tensor[start_f:end_f] = torch.tensor(row_labels, dtype=torch.long)
                    self.density_tensor[start_f:end_f] = d_class 
            
            audio_ptr += frames_per_measure
            
        logging.info("Beatmap Timeline constructed (Corrected BPM 181.685).")
        # Sliding Window Calculation (Input is 300 frames wide)
        self.num_samples = (self.num_frames - self.window_frames) // self.stride_frames + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Window: [idx*100 : idx*100 + 300]
        frame_start = idx * self.stride_frames
        input_end = frame_start + self.window_frames
        target_end = frame_start + self.frames_per_sec # Predict first 75 frames
        print("frame_start:   ",frame_start,"  input_end : ",input_end,"  target_end : " , target_end) # do not tuch this line
        
        # 1. INPUT (300 Frames -> 1200 Tokens)
        # CHECK: Have we reached the end of the token file?
        if input_end > self.num_frames:
            # YES: We are at the end.
            # PAD with zeros so we still have 300 frames.
            valid_len = self.num_frames - frame_start
            chunk = self.tokens_tensor[frame_start:self.num_frames]
            padding = torch.zeros((self.window_frames - valid_len, 4), dtype=torch.long)
            chunk = torch.cat([chunk, padding], dim=0)
        else:
            # NO: Grab the full 300 frames.
            chunk = self.tokens_tensor[frame_start:input_end]
            
        x = chunk.view(-1) # 1200 tokens
        
        # 2. TARGET (First 75 frames of the window)
        # Note: If near end, target might be short, need handling?
        # Assuming training data fits well for now or padding handled below
        target_valid_end = min(target_end, self.num_frames)
        y_dense = self.labels_tensor[frame_start:target_valid_end] 
        
        if target_valid_end > frame_start:
             y_d_class = self.density_tensor[frame_start:target_valid_end][0].item()
        else:
             y_d_class = 0
        
        d_map = {0: 4, 1: 8, 2: 12, 3: 16}
        num_lines = d_map.get(y_d_class, 16)
        
        step = 75 / num_lines
        indices = [int(i * step) for i in range(num_lines)]
        
        symbolic_notes = []
        for i in indices:
            if i < y_dense.shape[0]: 
                symbolic_notes.append(y_dense[i])
        
        if len(symbolic_notes) > 0:
            symbolic_tensor = torch.stack(symbolic_notes) 
        else:
            symbolic_tensor = torch.zeros((0, 4), dtype=torch.long)
            
        MAX_SLOTS = 16
        pad_len = MAX_SLOTS - symbolic_tensor.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, 4), dtype=torch.long)
            symbolic_tensor = torch.cat([symbolic_tensor, padding], dim=0)
            
        return x, symbolic_tensor, torch.tensor(y_d_class, dtype=torch.long), frame_start

# --- 2. Model: Flat Input LSTM ---
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

# --- 3. Training Loop ---
def train(model, dataloader, epochs=EPOCHS, device='cpu', song_name='unknown'):
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(device) 
    criterion_notes = nn.CrossEntropyLoss(weight=class_weights)
    criterion_density = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()
    
    logging.info(f"{'='*60}")
    logging.info(f"TRAINING START (300 Frame Window)")
    logging.info(f"Input: 300 Frames (1200 Tokens) -> Output: 1s Beatmap")
    logging.info(f"{'='*60}")
    
    best_loss = float('inf')
    best_epoch = 0
    epoch_losses = []  # Track loss for each epoch for graphing
    epoch_perplexities = []  # Track perplexity (exp of loss) for each epoch
    epoch_accuracies = []  # Track density accuracy for each epoch
    epoch_perfect_matches = []  # Track perfect beatmap match % for each epoch
    epoch_f1_scores = []  # Track macro F1-score for each epoch
    epoch_note_accuracies = []  # Track note-level accuracy for each epoch
    epoch_times = []  # Track training time per epoch
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Track epoch start time
        total_loss = 0
        epoch_density_correct = 0
        epoch_density_total = 0
        epoch_perfect_correct = 0
        epoch_perfect_total = 0
        
        # For F1-score calculation
        all_predictions = []
        all_targets = []
        epoch_note_correct = 0
        epoch_note_total = 0
        for batch_idx, (x, y_notes, y_density, frame_indices) in enumerate(dataloader):
            x, y_notes, y_density = x.to(device), y_notes.to(device), y_density.to(device)
            # x: (B, 1200)
            
            # --- LOGGING: Prove Sliding Window Logic ---
            # Log the range of frames being processed for the first few samples in the batch
            if batch_idx < 5: # Log first 5 batches to show progression
                # Get the first sample's start frame
                f_start = frame_indices[0].item()
                f_end = f_start + 300
                logging.info(f"Batch {batch_idx}: Processing Window [{f_start}:{f_end}] (Sample 0)")
            
            optimizer.zero_grad()
            notes_out, density_out = model(x) 
            
            # --- DEBUG: FULL INPUT LOGGING (Only if DEBUG_MODE is True) ---
            if DEBUG_MODE and epoch == 0:
                # Force full printing
                torch.set_printoptions(profile="full", linewidth=200)
                logging.info(f"\n{'='*20} BATCH {batch_idx+1} FULL INPUT & BEATMAP DUMP {'='*20}")
                
                # Loop through every sample in this batch
                batch_size = x.shape[0]
                for i in range(batch_size):
                    # Get associated frame index for clarity
                    f_start = frame_indices[i].item()
                    f_end = f_start + 300
                    
                    logging.info(f"--- Sample {i} (Frames {f_start}-{f_end}) ---")
                    
                    # Convert tensor to list and join with commas
                    tokens_list = x[i].tolist()
                    formatted_tokens = ",".join(map(str, tokens_list))
                    logging.info(f"INPUT (Tokens):\n{formatted_tokens}") 
                    
                    logging.info(f"TARGET BEATMAP (True Labels):\n{y_notes[i]}")
                
                logging.info(f"{'='*60}\n")
                # Reset
                torch.set_printoptions(profile="default")
            


            
            # --- VERIFICATION (Commented out for cleaner logs) ---
            # logging.info(f"--- BATCH {batch_idx+1}/{len(dataloader)} VERIFICATION ---")
            
            # for i in range(x.shape[0]):
            #     # ... (Detailed sample verification logic omitted for brevity) ...
            #     pass

            loss_n = criterion_notes(notes_out.reshape(-1, 5), y_notes.view(-1))
            loss_d = criterion_density(density_out, y_density)
            
            loss = loss_n + loss_d
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # PROVE COVERAGE: Log every batch
            # --- VALIDATION METRICS (User Request) ---
            # 1. Density Accuracy
            # density_out: [Batch, 4], y_density: [Batch] (Class Index)
            pred_density = torch.argmax(density_out, dim=1) 
            true_density = y_density # Already class indices
            density_matches = (pred_density == true_density).sum().item()
            
            # Track for epoch-level accuracy
            current_batch_size = x.size(0)  # Get actual batch size
            epoch_density_correct += density_matches
            epoch_density_total += current_batch_size

            
            # 2. Note Logic & Matches
            # notes_out: [Batch, 16, 4, 5], y_notes: [Batch, 16, 4]
            pred_notes_indices = torch.argmax(notes_out, dim=3) # [Batch, 16, 4] - argmax over the 5 note types
            
            # Count how many FULL 16-step sequences matched perfectly
            perfect_matches = 0
            batch_size = x.shape[0]
            for i in range(batch_size):
                if torch.equal(pred_notes_indices[i], y_notes[i]):
                    perfect_matches += 1
            
            # Track perfect matches for epoch average
            epoch_perfect_correct += perfect_matches
            epoch_perfect_total += current_batch_size
            
            # Calculate note-level accuracy (individual note positions)
            # Flatten predictions and targets for note-level metrics
            pred_flat = pred_notes_indices.cpu().numpy().flatten()
            target_flat = y_notes.cpu().numpy().flatten()
            
            # Track for F1-score (collect all predictions across batches)
            all_predictions.extend(pred_flat.tolist())
            all_targets.extend(target_flat.tolist())
            
            # Calculate note-level accuracy for this batch
            note_matches = (pred_flat == target_flat).sum()
            note_total = len(pred_flat)
            epoch_note_correct += note_matches
            epoch_note_total += note_total
            
            # Log summary for this batch
            logging.info(f"Batch {batch_idx+1} Validation:")
            logging.info(f"  Density Accuracy: {density_matches}/{current_batch_size} ({density_matches/current_batch_size:.0%}")
            logging.info(f"  Perfect Beatmap Matches: {perfect_matches}/{current_batch_size} ({perfect_matches/current_batch_size:.0%})")
            # --- END VALIDATION ---
            
            logging.info(f"Batch {batch_idx+1}/{len(dataloader)} Processed. (Loss: {loss.item():.4f})")
            
        avg_epoch_loss = total_loss / len(dataloader)

        if (epoch + 1) % 1 == 0:
             logging.info(f"Epoch {epoch+1}/{epochs} Completed. Avg Loss: {avg_epoch_loss:.4f}")
             
        if (epoch + 1) % 10 == 0:
            # Periodically verify one sample to ensure it's not deviating widely
            logging.info(f"--- Periodic Check (Epoch {epoch+1}) ---")
            # (Optional: Add single sample verification here if needed, but keeping it clean for now)
            # The batch logging below is redundant if the summary is added.
            # logging.info(f"Batch {batch_idx+1}/{len(dataloader)} Processed. (Loss: {loss.item():.4f})")
            
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)  # Save for graphing
        
        # Calculate perplexity (lower is better)
        epoch_ppl = math.exp(avg_epoch_loss) if avg_epoch_loss < 20 else float('inf')
        epoch_perplexities.append(epoch_ppl)

        # Calculate and store epoch accuracy
        epoch_accuracy = (epoch_density_correct / epoch_density_total * 100) if epoch_density_total > 0 else 0
        epoch_accuracies.append(epoch_accuracy)
        
        # Calculate and store epoch perfect match %
        epoch_perfect_pct = (epoch_perfect_correct / epoch_perfect_total * 100) if epoch_perfect_total > 0 else 0
        epoch_perfect_matches.append(epoch_perfect_pct)
        
        # Calculate F1-score (macro average across all note classes: 0,1,2,3,4)
        if len(all_predictions) > 0 and len(all_targets) > 0:
            epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
            epoch_f1_scores.append(epoch_f1)
        else:
            epoch_f1_scores.append(0.0)
            epoch_f1 = 0.0
        
        # Calculate note-level accuracy
        epoch_note_acc = (epoch_note_correct / epoch_note_total * 100) if epoch_note_total > 0 else 0
        epoch_note_accuracies.append(epoch_note_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate per-direction precision and recall
        direction_labels = {1: 'Left', 2: 'Down', 3: 'Up', 4: 'Right'}
        direction_metrics = {}
        for dir_val, dir_name in direction_labels.items():
            # Create binary classification: this direction vs all others
            y_true_binary = [1 if t == dir_val else 0 for t in all_targets]
            y_pred_binary = [1 if p == dir_val else 0 for p in all_predictions]
            
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            direction_metrics[dir_name] = {'precision': prec, 'recall': rec}
        
        logging.info(f"Epoch {epoch+1}/{epochs} Summary:")
        logging.info(f"  Total Samples Processed: {epoch_density_total}") # Using epoch_density_total as samples_in_epoch
        logging.info(f"  Average Loss: {avg_epoch_loss:.4f}")
        logging.info(f"  Perplexity (PPL): {epoch_ppl:.4f}")
        logging.info(f"  Density Accuracy: {epoch_accuracy:.1f}%")
        logging.info(f"  Perfect Match %: {epoch_perfect_pct:.1f}%")
        logging.info(f"  F1-Score (Macro): {epoch_f1:.1f}%")
        logging.info(f"  Note-Level Accuracy: {epoch_note_acc:.1f}%")
        logging.info(f"  Epoch Time: {epoch_time:.2f}s")
        logging.info(f"  Per-Direction Metrics:")
        for dir_name, metrics in direction_metrics.items():
            logging.info(f"    {dir_name}: Precision={metrics['precision']:.1f}%, Recall={metrics['recall']:.1f}%")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "src/training/beatmap_lstm_springtime_best.pth")
            logging.info(f"  ✓ New best model saved! (Epoch {best_epoch})")
    
    # --- GENERATE DUAL-AXIS GRAPH ---
    logging.info("Generating loss & accuracy graph...")
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot Loss and PPL on left axis
    color_loss = '#2563eb'
    color_ppl = '#dc2626'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss / Perplexity', fontsize=12)
    line1 = ax1.plot(range(1, epochs+1), epoch_losses, linewidth=2, color=color_loss, label='Loss', marker='o', markersize=3)
    line_ppl = ax1.plot(range(1, epochs+1), epoch_perplexities, linewidth=2, color=color_ppl, label='Perplexity', marker='x', markersize=4, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for Accuracies
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy / Score (%)', fontsize=12)
    
    # Plot all accuracy metrics on right axis
    line2 = ax2.plot(range(1, epochs+1), epoch_accuracies, linewidth=2, color='#10b981', 
                     label='Density Acc', marker='o', markersize=4)
    line3 = ax2.plot(range(1, epochs+1), epoch_perfect_matches, linewidth=2, color='#f59e0b', 
                     label='Perfect Match %', marker='s', markersize=4)
    line4 = ax2.plot(range(1, epochs+1), epoch_f1_scores, linewidth=2, color='#8b5cf6', 
                     label='F1-Score', marker='^', markersize=4)
    line5 = ax2.plot(range(1, epochs+1), epoch_note_accuracies, linewidth=2, color='#ec4899', 
                     label='Note-Level Acc', marker='d', markersize=4)
    
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 105)  # Accuracy from 0-100%
    
    # Add title and legend
    plt.title('Training Metrics: Comprehensive View', fontsize=14, fontweight='bold')
    
    # Combine legends from both axes
    lines = line1 + line_ppl + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, ncol=3)
    
    # Mark best epoch with vertical line
    if best_epoch > 0:
        ax1.axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
        # Add annotation for best epoch
        ax1.annotate(f'★ Best\n(Epoch {best_epoch})', 
                    xy=(best_epoch, epoch_losses[best_epoch-1]), 
                    xytext=(best_epoch + 1, epoch_losses[best_epoch-1] * 1.5),
                    fontsize=8, color='gold', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gold', lw=1.5))
    
    # Annotate final values on line endpoints
    if epochs > 0:
        # Loss final value
        ax1.text(epochs, epoch_losses[-1], f' {epoch_losses[-1]:.4f}', 
                fontsize=8, color='#2563eb', va='center', fontweight='bold')
        # PPL final value  
        ax1.text(epochs, epoch_perplexities[-1], f' {epoch_perplexities[-1]:.3f}', 
                fontsize=8, color='#dc2626', va='center', fontweight='bold')
        # Accuracy final values
        ax2.text(epochs, epoch_accuracies[-1], f' {epoch_accuracies[-1]:.1f}%', 
                fontsize=8, color='#10b981', va='center', fontweight='bold')
        ax2.text(epochs, epoch_perfect_matches[-1], f' {epoch_perfect_matches[-1]:.1f}%', 
                fontsize=8, color='#f59e0b', va='center', fontweight='bold')
        ax2.text(epochs, epoch_f1_scores[-1], f' {epoch_f1_scores[-1]:.1f}%', 
                fontsize=8, color='#8b5cf6', va='center', fontweight='bold')
        ax2.text(epochs, epoch_note_accuracies[-1], f' {epoch_note_accuracies[-1]:.1f}%', 
                fontsize=8, color='#ec4899', va='center', fontweight='bold')
    
    # Add final metrics annotation with all metrics + timing
    if epoch_losses and epoch_accuracies:
        final_loss = epoch_losses[-1]
        final_ppl = epoch_perplexities[-1]
        final_acc = epoch_accuracies[-1]
        final_perfect = epoch_perfect_matches[-1]
        final_f1 = epoch_f1_scores[-1]
        final_note_acc = epoch_note_accuracies[-1]
        total_time = sum(epoch_times)
        avg_time = total_time / len(epoch_times) if len(epoch_times) > 0 else 0
        
        textstr = (f'Final Metrics:\n'
                  f'Loss: {final_loss:.4f}\n'
                  f'PPL: {final_ppl:.4f}\n'
                  f'Density: {final_acc:.1f}%\n'
                  f'Perfect: {final_perfect:.1f}%\n'
                  f'F1: {final_f1:.1f}%\n'
                  f'Note: {final_note_acc:.1f}%\n'
                  f'---\n'
                  f'Best: Ep.{best_epoch}\n'
                  f'Time: {total_time:.1f}s\n'
                  f'Avg: {avg_time:.1f}s/ep')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = f"src/training/loss_graph_{song_name}_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✓ Loss & Accuracy graph saved to: {graph_path}")
            
    return model

# --- 4. Training with Weights & Biases (W&B) ---
def train_with_wandb(model, dataloader, epochs=EPOCHS, device='cpu', song_name='unknown', 
                     project_name='beatmap-lstm', run_name=None, learning_rate=1e-3):
    """
    Train with Weights & Biases for beautiful interactive graphs!
    
    Features:
    - Real-time cloud dashboard with interactive graphs
    - Confusion matrices for notes and density
    - Per-direction precision/recall tracking
    - Automatic model checkpointing
    - Shareable results via URL
    
    Install: pip install wandb
    Login: wandb login (one-time setup)
    """
    try:
        import wandb
    except ImportError:
        logging.error("❌ wandb not installed. Run: pip install wandb")
        logging.error("Falling back to regular training...")
        return train(model, dataloader, epochs, device, song_name)
    
    # Initialize W&B
    if run_name is None:
        run_name = f"{song_name}_lstm_{time.strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': dataloader.batch_size,
            'device': str(device),
            'song_name': song_name,
        }
    )
    
    # Watch model
    wandb.watch(model, log='all', log_freq=100)
    
    # Setup training
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion_notes = nn.CrossEntropyLoss(weight=class_weights)
    criterion_density = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    logging.info(f"{'='*60}")
    logging.info(f"TRAINING WITH WANDB")
    logging.info(f"🔗 Dashboard: {wandb.run.get_url()}")
    logging.info(f"{'='*60}")
    
    best_loss = float('inf')
    best_epoch = 0
    direction_labels = {1: 'Left', 2: 'Down', 3: 'Up', 4: 'Right'}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        total_loss_notes = 0
        total_loss_density = 0
        epoch_density_correct = 0
        epoch_density_total = 0
        epoch_perfect_correct = 0
        epoch_perfect_total = 0
        epoch_note_correct = 0
        epoch_note_total = 0
        
        all_predictions = []
        all_targets = []
        all_density_preds = []
        all_density_targets = []
        all_concurrency_preds = []  # Track note concurrency (0/1/2/3/4 simultaneous notes)
        all_concurrency_targets = []
        
        for batch_idx, (x, y_notes, y_density, frame_indices) in enumerate(dataloader):
            x, y_notes, y_density = x.to(device), y_notes.to(device), y_density.to(device)
            
            optimizer.zero_grad()
            notes_out, density_out = model(x)
            
            loss_n = criterion_notes(notes_out.reshape(-1, 5), y_notes.view(-1))
            loss_d = criterion_density(density_out, y_density)
            loss = loss_n + loss_d
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_notes += loss_n.item()
            total_loss_density += loss_d.item()
            
            # Metrics
            current_batch_size = x.size(0)
            pred_density = torch.argmax(density_out, dim=1)
            density_matches = (pred_density == y_density).sum().item()
            epoch_density_correct += density_matches
            epoch_density_total += current_batch_size
            
            all_density_preds.extend(pred_density.cpu().numpy().tolist())
            all_density_targets.extend(y_density.cpu().numpy().tolist())
            
            pred_notes_indices = torch.argmax(notes_out, dim=3)
            perfect_matches = sum(torch.equal(pred_notes_indices[i], y_notes[i]) 
                                for i in range(current_batch_size))
            epoch_perfect_correct += perfect_matches
            epoch_perfect_total += current_batch_size
            
            pred_flat = pred_notes_indices.cpu().numpy().flatten()
            target_flat = y_notes.cpu().numpy().flatten()
            all_predictions.extend(pred_flat.tolist())
            all_targets.extend(target_flat.tolist())
            
            # Track concurrency (simultaneous note count per line)
            for i in range(current_batch_size):
                for frame_idx in range(pred_notes_indices.shape[1]):  # 75 frames
                    pred_line = pred_notes_indices[i, frame_idx].cpu().numpy()
                    target_line = y_notes[i, frame_idx].cpu().numpy()
                    
                    # Count non-zero notes (0=empty, 1/2/3/4=note types)
                    pred_concurrency = sum(1 for note in pred_line if note != 0)
                    target_concurrency = sum(1 for note in target_line if note != 0)
                    
                    all_concurrency_preds.append(pred_concurrency)
                    all_concurrency_targets.append(target_concurrency)
            
            note_matches = (pred_flat == target_flat).sum()
            epoch_note_correct += note_matches
            epoch_note_total += len(pred_flat)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_loss_notes = total_loss_notes / len(dataloader)
        avg_loss_density = total_loss_density / len(dataloader)
        epoch_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        epoch_density_acc = (epoch_density_correct / epoch_density_total * 100) if epoch_density_total > 0 else 0
        epoch_perfect_pct = (epoch_perfect_correct / epoch_perfect_total * 100) if epoch_perfect_total > 0 else 0
        epoch_note_acc = (epoch_note_correct / epoch_note_total * 100) if epoch_note_total > 0 else 0
        epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
        epoch_time = time.time() - epoch_start
        
        # Per-direction metrics
        direction_metrics = {}
        for dir_val, dir_name in direction_labels.items():
            y_true_bin = [1 if t == dir_val else 0 for t in all_targets]
            y_pred_bin = [1 if p == dir_val else 0 for p in all_predictions]
            prec = precision_score(y_true_bin, y_pred_bin, zero_division=0) * 100
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0) * 100
            direction_metrics[dir_name] = {'precision': prec, 'recall': rec}
        
        # Log to W&B
        wandb_log = {
            'epoch': epoch + 1,
            'loss/total': avg_loss,
            'loss/notes': avg_loss_notes,
            'loss/density': avg_loss_density,
            'perplexity': epoch_ppl,
            'accuracy/density': epoch_density_acc,
            'accuracy/perfect_match': epoch_perfect_pct,
            'accuracy/note_level': epoch_note_acc,
            'f1_score': epoch_f1,
            'time/epoch_seconds': epoch_time,
        }
        
        for dir_name, metrics in direction_metrics.items():
            wandb_log[f'direction/{dir_name.lower()}/precision'] = metrics['precision']
            wandb_log[f'direction/{dir_name.lower()}/recall'] = metrics['recall']
        
        wandb.log(wandb_log)
        
        # Log confusion matrices every 10 epochs
        if (epoch + 1) % 10 == 0:
            wandb.log({
                'confusion_matrix/notes': wandb.plot.confusion_matrix(
                    probs=None, y_true=all_targets, preds=all_predictions,
                    class_names=['Empty', 'Left', 'Down', 'Up', 'Right']
                ),
                'confusion_matrix/density': wandb.plot.confusion_matrix(
                    probs=None, y_true=all_density_targets, preds=all_density_preds,
                    class_names=['Low (≤4)', 'Med (5-8)', 'High (9-12)', 'VHigh (13+)']
                ),
                'confusion_matrix/concurrency': wandb.plot.confusion_matrix(
                    probs=None, y_true=all_concurrency_targets, preds=all_concurrency_preds,
                    class_names=['Empty (0)', 'Single (1)', 'Double (2)', 'Triple (3)', 'Quad (4)']
                )
            })
        
        # Console output
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                    f"Density: {epoch_density_acc:.1f}% | Perfect: {epoch_perfect_pct:.1f}% | "
                    f"F1: {epoch_f1:.1f}% | Time: {epoch_time:.2f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            save_path = f"src/training/beatmap_lstm_{song_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_loss'] = best_loss
            logging.info(f"  ✓ New best model saved!")
    
    # Save final model
    final_path = f"src/training/beatmap_lstm_{song_name}_final.pth"
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Training Complete! Best: Epoch {best_epoch}, Loss {best_loss:.4f}")
    logging.info(f"View results: {wandb.run.get_url()}")
    logging.info(f"{'='*60}")
    
    wandb.finish()
    return model


if __name__ == "__main__":
    setup_logging()
    token_file = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv"
    ssc_file = "src/musicForBeatmap/Springtime/beatmap_easy.text"
    
    # Extract song name from token file
    import os
    token_basename = os.path.basename(token_file)
    # Extract name between "tokens_" and timestamp
    if "tokens_" in token_basename:
        song_name = token_basename.split("tokens_")[1].split("_202")[0]
    else:
        song_name = "unknown"
    
    try:
        ds = SpringtimeDataset(token_csv_path=token_file, ssc_path=ssc_file)
        
        # Display FIRST sample for debugging
        x_val, y_notes_val, y_dens_val, frame_val = ds[0]
        print(f"\n{'='*60}")
        print(f"SPRINGTIME DATASET: FIRST SAMPLE")
        print(f"{'='*60}")
        print(f"Input Shape (Flat Tokens): {x_val.shape}")
        print(f"Target Notes Shape: {y_notes_val.shape}")
        print(f"Target Density (Class Index): {y_dens_val.item() if hasattr(y_dens_val, 'item') else y_dens_val}")
        print(f"Frame Start Index: {frame_val.item() if hasattr(frame_val, 'item') else frame_val}")
        print(f"First 16 Tokens (4 Frames):\n{x_val[:16]}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logging.critical(f"Error: {e}")
        exit(1)
    
    
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    model = BeatmapLSTM()
    
    # --- RESUME TRAINING: Load existing weights if available ---
    best_weights_path = "src/training/beatmap_lstm_springtime_best.pth"
    if os.path.exists(best_weights_path):
        logging.info(f"✓ Loading existing weights from: {best_weights_path}")
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
        logging.info("✓ Weights loaded. Training will continue from where it left off.")
    else:
        logging.info("No existing weights found. Starting from scratch.")
    
    if torch.backends.mps.is_available(): 
        model.to('mps')
    
    # ============================================================
    # CHOOSE YOUR TRAINING METHOD:
    # ============================================================
    
    # Option 1: Regular training with PNG graphs
    # train(model, loader, epochs=EPOCHS, 
    #       device='mps' if torch.backends.mps.is_available() else 'cpu', 
    #       song_name=song_name)
    
    # Option 2: W&B training with interactive cloud dashboards (ACTIVE!)
    # Prerequisites: pip install wandb && wandb login
    train_with_wandb(model, loader, epochs=EPOCHS, 
                     device='mps' if torch.backends.mps.is_available() else 'cpu', 
                     song_name=song_name, 
                     project_name='beatmap-lstm')
    
    torch.save(model.state_dict(), "src/training/beatmap_lstm_springtime.pth")
    logging.info("Training Complete.")

