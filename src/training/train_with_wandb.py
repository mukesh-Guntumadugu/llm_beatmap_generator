"""
Training Script with Weights & Biases Integration
==================================================
This module provides a comprehensive training function that uses Weights & Biases (wandb)
to create beautiful, interactive graphs and track all training metrics in real-time.

Features:
- Real-time metric visualization on wandb dashboard
- Interactive graphs for loss, accuracy, F1-score, and more
- Automatic hyperparameter tracking
- Model checkpoint management
- Per-direction precision/recall tracking
- Confusion matrix logging
- Hardware utilization monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import time
import math
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

# Import from main training file
from train import SpringtimeDataset, BeatmapLSTM


def train_with_wandb(
    model,
    dataloader,
    epochs=300,
    device='cpu',
    song_name='unknown',
    project_name='beatmap-lstm',
    run_name=None,
    learning_rate=1e-3,
    save_dir='src/training',
    log_interval=10
):
    """
    Train the beatmap LSTM model with Weights & Biases integration.
    
    This function provides comprehensive metric tracking and beautiful visualizations
    through the wandb dashboard, including:
    - Loss and perplexity curves
    - Accuracy metrics (density, note-level, perfect match)
    - F1-scores and per-direction metrics
    - Training time analytics
    - Confusion matrices
    - Model checkpoints
    
    Args:
        model: BeatmapLSTM model to train
        dataloader: PyTorch DataLoader with training data
        epochs: Number of training epochs (default: 300)
        device: Device to train on ('cpu', 'cuda', or 'mps')
        song_name: Name of the song being trained on
        project_name: W&B project name
        run_name: W&B run name (auto-generated if None)
        learning_rate: Learning rate for Adam optimizer
        save_dir: Directory to save model checkpoints
        log_interval: How often to log detailed metrics (in epochs)
    
    Returns:
        Trained model
    """
    
    # Initialize Weights & Biases
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
            'model_type': 'BeatmapLSTM',
            'input_frames': 300,
            'output_frames': 75,
            'embed_dim': model.embedding.embedding_dim,
            'hidden_dim': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers,
        }
    )
    
    # Watch model gradients and parameters
    wandb.watch(model, log='all', log_freq=100)
    
    # Setup loss functions
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion_notes = nn.CrossEntropyLoss(weight=class_weights)
    criterion_density = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    logging.info(f"{'='*60}")
    logging.info(f"TRAINING WITH WANDB (Project: {project_name}, Run: {run_name})")
    logging.info(f"Input: 300 Frames (1200 Tokens) -> Output: 1s Beatmap")
    logging.info(f"{'='*60}")
    logging.info(f"🔗 View your training dashboard at: {wandb.run.get_url()}")
    
    best_loss = float('inf')
    best_epoch = 0
    
    # Direction labels for per-direction metrics
    direction_labels = {1: 'Left', 2: 'Down', 3: 'Up', 4: 'Right'}
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Reset metrics for this epoch
        total_loss = 0
        total_loss_notes = 0
        total_loss_density = 0
        epoch_density_correct = 0
        epoch_density_total = 0
        epoch_perfect_correct = 0
        epoch_perfect_total = 0
        epoch_note_correct = 0
        epoch_note_total = 0
        
        # Collect all predictions and targets for F1-score
        all_predictions = []
        all_targets = []
        all_density_preds = []
        all_density_targets = []
        
        for batch_idx, (x, y_notes, y_density, frame_indices) in enumerate(dataloader):
            x, y_notes, y_density = x.to(device), y_notes.to(device), y_density.to(device)
            
            optimizer.zero_grad()
            notes_out, density_out = model(x)
            
            # Calculate losses
            loss_n = criterion_notes(notes_out.reshape(-1, 5), y_notes.view(-1))
            loss_d = criterion_density(density_out, y_density)
            loss = loss_n + loss_d
            
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_loss_notes += loss_n.item()
            total_loss_density += loss_d.item()
            
            # Calculate batch metrics
            current_batch_size = x.size(0)
            
            # Density accuracy
            pred_density = torch.argmax(density_out, dim=1)
            density_matches = (pred_density == y_density).sum().item()
            epoch_density_correct += density_matches
            epoch_density_total += current_batch_size
            
            # Collect for confusion matrix
            all_density_preds.extend(pred_density.cpu().numpy().tolist())
            all_density_targets.extend(y_density.cpu().numpy().tolist())
            
            # Note predictions
            pred_notes_indices = torch.argmax(notes_out, dim=3)
            
            # Perfect matches
            perfect_matches = sum(
                torch.equal(pred_notes_indices[i], y_notes[i])
                for i in range(current_batch_size)
            )
            epoch_perfect_correct += perfect_matches
            epoch_perfect_total += current_batch_size
            
            # Note-level accuracy
            pred_flat = pred_notes_indices.cpu().numpy().flatten()
            target_flat = y_notes.cpu().numpy().flatten()
            all_predictions.extend(pred_flat.tolist())
            all_targets.extend(target_flat.tolist())
            
            note_matches = (pred_flat == target_flat).sum()
            epoch_note_correct += note_matches
            epoch_note_total += len(pred_flat)
            
            # Log batch metrics to wandb (every N batches to avoid overwhelming)
            if batch_idx % 10 == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/loss_notes': loss_n.item(),
                    'batch/loss_density': loss_d.item(),
                    'batch/density_accuracy': density_matches / current_batch_size * 100,
                })
        
        # Calculate epoch metrics
        avg_epoch_loss = total_loss / len(dataloader)
        avg_loss_notes = total_loss_notes / len(dataloader)
        avg_loss_density = total_loss_density / len(dataloader)
        
        epoch_ppl = math.exp(avg_epoch_loss) if avg_epoch_loss < 20 else float('inf')
        epoch_density_acc = (epoch_density_correct / epoch_density_total * 100) if epoch_density_total > 0 else 0
        epoch_perfect_pct = (epoch_perfect_correct / epoch_perfect_total * 100) if epoch_perfect_total > 0 else 0
        epoch_note_acc = (epoch_note_correct / epoch_note_total * 100) if epoch_note_total > 0 else 0
        
        # Calculate F1-score
        if len(all_predictions) > 0 and len(all_targets) > 0:
            epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
        else:
            epoch_f1 = 0.0
        
        # Calculate per-direction metrics
        direction_metrics = {}
        for dir_val, dir_name in direction_labels.items():
            y_true_binary = [1 if t == dir_val else 0 for t in all_targets]
            y_pred_binary = [1 if p == dir_val else 0 for p in all_predictions]
            
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0) * 100
            
            direction_metrics[dir_name] = {
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch metrics to wandb
        wandb_log = {
            'epoch': epoch + 1,
            'loss/total': avg_epoch_loss,
            'loss/notes': avg_loss_notes,
            'loss/density': avg_loss_density,
            'perplexity': epoch_ppl,
            'accuracy/density': epoch_density_acc,
            'accuracy/perfect_match': epoch_perfect_pct,
            'accuracy/note_level': epoch_note_acc,
            'f1_score/macro': epoch_f1,
            'time/epoch_seconds': epoch_time,
        }
        
        # Add per-direction metrics
        for dir_name, metrics in direction_metrics.items():
            wandb_log[f'direction/{dir_name.lower()}/precision'] = metrics['precision']
            wandb_log[f'direction/{dir_name.lower()}/recall'] = metrics['recall']
            wandb_log[f'direction/{dir_name.lower()}/f1'] = metrics['f1']
        
        wandb.log(wandb_log)
        
        # Log confusion matrix periodically
        if (epoch + 1) % log_interval == 0:
            # Note confusion matrix
            cm_notes = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3, 4])
            wandb.log({
                'confusion_matrix/notes': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_targets,
                    preds=all_predictions,
                    class_names=['Empty', 'Left', 'Down', 'Up', 'Right']
                )
            })
            
            # Density confusion matrix
            wandb.log({
                'confusion_matrix/density': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_density_targets,
                    preds=all_density_preds,
                    class_names=['4 lines', '8 lines', '12 lines', '16 lines']
                )
            })
        
        # Print progress
        logging.info(f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {avg_epoch_loss:.4f} | "
                    f"PPL: {epoch_ppl:.4f} | "
                    f"Density: {epoch_density_acc:.1f}% | "
                    f"Perfect: {epoch_perfect_pct:.1f}% | "
                    f"F1: {epoch_f1:.1f}% | "
                    f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            
            # Save to local directory
            best_model_path = os.path.join(save_dir, f"beatmap_lstm_{song_name}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            
            # Save to wandb
            wandb.save(best_model_path)
            
            # Log best metrics
            wandb.run.summary['best_epoch'] = best_epoch
            wandb.run.summary['best_loss'] = best_loss
            wandb.run.summary['best_density_acc'] = epoch_density_acc
            wandb.run.summary['best_f1'] = epoch_f1
            
            logging.info(f"  ✓ New best model saved! (Epoch {best_epoch})")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"beatmap_lstm_{song_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    
    # Finish wandb run
    logging.info(f"\n{'='*60}")
    logging.info(f"Training Complete!")
    logging.info(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")
    logging.info(f"View full results at: {wandb.run.get_url()}")
    logging.info(f"{'='*60}")
    
    wandb.finish()
    
    return model


if __name__ == "__main__":
    """
    Example usage of the wandb training function.
    
    Before running this script, make sure to:
    1. Install wandb: pip install wandb
    2. Login to wandb: wandb login
    3. Adjust the paths and parameters as needed
    """
    
    import logging
    from train import setup_logging
    
    # Setup logging
    setup_logging()
    
    # Configuration
    token_file = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv"
    ssc_file = "src/musicForBeatmap/Springtime/beatmap_easy.text"
    
    # Extract song name
    token_basename = os.path.basename(token_file)
    if "tokens_" in token_basename:
        song_name = token_basename.split("tokens_")[1].split("_202")[0]
    else:
        song_name = "unknown"
    
    # Create dataset
    try:
        ds = SpringtimeDataset(token_csv_path=token_file, ssc_path=ssc_file)
        logging.info(f"Dataset created with {len(ds)} samples")
    except Exception as e:
        logging.critical(f"Error creating dataset: {e}")
        exit(1)
    
    # Create dataloader
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # Create model
    model = BeatmapLSTM()
    
    # Check for existing weights
    best_weights_path = f"src/training/beatmap_lstm_{song_name}_best.pth"
    if os.path.exists(best_weights_path):
        logging.info(f"✓ Loading existing weights from: {best_weights_path}")
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
        logging.info("✓ Weights loaded.")
    else:
        logging.info("Starting from scratch - no existing weights found.")
    
    # Select device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    logging.info(f"Using device: {device}")
    
    # Train with wandb
    train_with_wandb(
        model=model,
        dataloader=loader,
        epochs=300,
        device=device,
        song_name=song_name,
        project_name='beatmap-lstm-training',
        learning_rate=1e-3,
        save_dir='src/training',
        log_interval=10
    )
    
    logging.info("Training complete!")
