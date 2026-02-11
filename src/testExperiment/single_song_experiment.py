import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
import time
import numpy as np

# Add src to path to allow imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

# Import existing modules
# We need to use importlib for modules with spaces or complex paths if direct import fails,
# but let's try to add the subdirectories to path first.
sys.path.append(str(src_dir / "Neural Audio Codecs"))
sys.path.append(str(src_dir / "training"))
sys.path.append(str(src_dir / "generating_of_beatmap"))

# Import dataset processing functions
from dataset_processing import create_aligned_dataset

# --- Define Model Locally to ensure consistency ---
class BeatmapLSTM(nn.Module):
    def __init__(self, num_codebooks=32, codebook_size=1024, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(num_codebooks)
        ])
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0) # No dropout for overfitting
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

class SingleSongDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.tokens = data['tokens'].long() # (32, T)
        self.targets = data['targets'].long() # (T, 4)
        
        # Transpose tokens to (T, 32)
        if self.tokens.shape[0] == 32:
            self.tokens = self.tokens.t()
            
        # Ensure length matches
        min_len = min(self.tokens.shape[0], self.targets.shape[0])
        self.tokens = self.tokens[:min_len]
        self.targets = self.targets[:min_len]
        
    def __len__(self):
        # We just return 1 "sequence" that is the whole song for this experiment
        # Or simpler: return 1 and getitem returns the whole tensor unsqueezed
        return 1
    
    def __getitem__(self, idx):
        # Return full song sequence
        # (T, 32), (T, 4)
        return self.tokens, self.targets

def train_single_song(dataset_path, model_save_path, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataset
    ds = SingleSongDataset(dataset_path)
    # We don't really need a dataloader for 1 item but it handles batch dim
    loader = DataLoader(ds, batch_size=1)
    
    model = BeatmapLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Higher LR for overfitting
    
    # Loss: Weighted to prioritize notes over empty space (0)
    # 0=Empty, 1=Tap, 2=Hold, 3=Tail, 4=Roll, 5=Mine
    weights = torch.tensor([0.05, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # x: (1, T, 32)
            # y: (1, T, 4)
            
            optimizer.zero_grad()
            logits = model(x) # (1, T, 4, 6)
            
            # Flatten for loss
            loss = criterion(logits.view(-1, 6), y.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.5f}")
            
    print(f"Training finished in {time.time()-start_time:.1f}s")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

def generate_and_verify(model, dataset_path, output_txt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load input tokens again
    ds = SingleSongDataset(dataset_path)
    tokens, targets = ds[0] # (T, 32), (T, 4)
    tokens = tokens.unsqueeze(0).to(device) # (1, T, 32)
    
    model.eval()
    with torch.no_grad():
        logits = model(tokens) # (1, T, 4, 6)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).squeeze(0).cpu() # (T, 4)
        
    # Stats
    total_frames = preds.shape[0]
    total_notes = (preds > 0).sum().item()
    
    print("\n--- Generation Stats ---")
    print(f"Total Frames: {total_frames}")
    print(f"Total Non-Empty Notes Generated: {total_notes}")
    print(f"Note Density: {total_notes/total_frames:.4f} notes/frame")
    
    # Compare with Ground Truth
    gt_notes = (targets > 0).sum().item()
    print(f"Ground Truth Non-Empty Notes: {gt_notes}")
    
    # Variance Check: Are we generating different notes?
    unique_rows = set()
    for i in range(preds.shape[0]):
        row_tuple = tuple(preds[i].tolist())
        unique_rows.add(row_tuple)
        
    print(f"Unique Row Patterns: {len(unique_rows)}")
    
    # Save to file
    int_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: 'M'}
    lines = []
    
    # Group by 75 frames (approx 1 sec)
    frames_per_measure = 75
    
    for i in range(total_frames):
        row_str = "".join([int_map[val.item()] for val in preds[i]])
        lines.append(row_str)
        
        if (i + 1) % frames_per_measure == 0 and (i + 1) < total_frames:
            lines.append(",")
            
    with open(output_txt_path, 'w') as f:
        f.write("\n".join(lines))
        f.write(";")
        
    print(f"Generated chart saved to {output_txt_path}")

def run_experiment():
    # Paths
    base_dir = Path("src")
    music_dir = base_dir / "musicForBeatmap/Springtime"
    ssc_path = music_dir / "Springtime.ssc"
    audio_path = music_dir / "Kommisar - Springtime.mp3"
    
    # Outputs in testExperiment
    exp_dir = Path("src/testExperiment")
    exp_dir.mkdir(exist_ok=True)
    
    tokens_path = exp_dir / "Springtime_tokens.pt"
    dataset_path = exp_dir / "Springtime_dataset.pt"
    model_path = exp_dir / "overfit_model.pth"
    generated_chart_path = exp_dir / "Springtime_generated.txt"
    
    print("=== Step 1: Tokenization ===")
    if not tokens_path.exists():
        print("Tokenizing audio (this might take a moment)...")
        # Dynamic import for EnCodec
        import importlib.util
        spec = importlib.util.spec_from_file_location("EnCodecimplementation", base_dir / "Neural Audio Codecs/EnCodecimplementation.py")
        enc = importlib.util.module_from_spec(spec)
        sys.modules["EnCodecimplementation"] = enc
        spec.loader.exec_module(enc)
        
        tokenizer = enc.AudioTokenizer(device='cpu', target_bandwidth=24.0)
        tokens = tokenizer.tokenize(str(audio_path))
        torch.save(tokens, tokens_path)
        print(f"Tokens saved to {tokens_path}")
    else:
        print(f"Using existing tokens: {tokens_path}")
        
    print("\n=== Step 2: Dataset Creation ===")
    if not dataset_path.exists():
        create_aligned_dataset(ssc_path, tokens_path, dataset_path)
    else:
        print(f"Using existing dataset: {dataset_path}")
        
    print("\n=== Step 3: Training (Overfitting) ===")
    model = train_single_song(dataset_path, model_path, epochs=150)
    
    print("\n=== Step 4: Generation & Verification ===")
    generate_and_verify(model, dataset_path, generated_chart_path)
    
if __name__ == "__main__":
    run_experiment()
