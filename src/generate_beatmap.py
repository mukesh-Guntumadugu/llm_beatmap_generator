import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np

# Re-define the model class (must match training exactly)
class BeatmapLSTM(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128, hidden_dim=256, output_dim=4, num_layers=2):
        super(BeatmapLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

def generate(token_path, model_path, output_txt_path, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = BeatmapLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Tokens
    print(f"Loading tokens from {token_path}...")
    tokens = torch.load(token_path, map_location=device)
    # Shape (1, 8, Time) -> (1, Time) (Codebook 0)
    input_tokens = tokens[:, 0, :].long()
    
    # 3. Predict
    print("Generating beatmap...")
    with torch.no_grad():
        logits = model(input_tokens) # (1, Time, 4)
        probabilities = torch.sigmoid(logits)
        
    # 4. Process Output
    probs = probabilities.squeeze(0).cpu().numpy() # (Time, 4)
    
    # Analyze predictions
    active_beats = []
    frame_rate = 75.0
    
    print(f"\n--- Analysis (Threshold > {threshold}) ---")
    
    with open(output_txt_path, 'w') as f:
        f.write(f"Generated Beatmap for: {Path(token_path).stem}\n")
        f.write(f"Model: {Path(model_path).name}\n")
        f.write("-" * 40 + "\n")
        
        for t in range(probs.shape[0]):
            step_probs = probs[t]
            # Check if any arrow is above threshold
            arrows = []
            if step_probs[0] > threshold: arrows.append("LEFT")
            if step_probs[1] > threshold: arrows.append("DOWN")
            if step_probs[2] > threshold: arrows.append("UP")
            if step_probs[3] > threshold: arrows.append("RIGHT")
            
            if arrows:
                time_sec = t / frame_rate
                # Format: 0023.45s: [LEFT, RIGHT] (Prob: 0.89)
                line = f"{time_sec:07.2f}s | Token {t:05d} | {' + '.join(arrows)}"
                print(line)
                f.write(line + "\n")
                active_beats.append(t)
                
    print(f"\nTotal beats generated: {len(active_beats)}")
    print(f"Output saved to {output_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("token_path", help="Path to input .pt token file")
    parser.add_argument("--model", help="Path to .pth model file", default="outputs/beatmap_lstm_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for a beat")
    
    args = parser.parse_args()
    
    # Deduce output path
    p = Path(args.token_path)
    out_txt = p.parent / f"{p.stem}_generated.txt"
    
    generate(args.token_path, args.model, out_txt, args.threshold)
