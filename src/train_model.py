import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import time

class BeatmapLSTM(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128, hidden_dim=256, output_dim=4, num_layers=2):
        super(BeatmapLSTM, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # LSTM Layer
        # batch_first=True makes input shape (Batch, Seq, Features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output Layer (maps hidden state to 4 standard arrows)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, SequenceLength) - Input is integer tokens
        
        # 1. Embed
        # shape: (Batch, Seq, EmbDim)
        embedded = self.embedding(x)
        
        # 2. LSTM
        # output shape: (Batch, Seq, HiddenDim)
        # (h_n, c_n) are final states, we don't need them for sequence prediction
        lstm_out, _ = self.lstm(embedded)
        
        # 3. Linear
        # shape: (Batch, Seq, 4)
        logits = self.fc(lstm_out)
        
        return logits

def train(dataset_path, epochs=100):
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)
    
    # Extract Inputs and Labels
    # Tokens: shape (1, 8, Time). We take codebook 0 for now.
    # shape becomes (1, Time)
    tokens = dataset['audio_tokens'][:, 0, :].long().to(device)
    
    # Labels: shape (Time, 4). We need to add batch dimension -> (1, Time, 4)
    labels = dataset['beatmap_labels'].unsqueeze(0).to(device)
    
    # 2. Initialize Model
    model = BeatmapLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Binary Cross Entropy with Logits (more stable than Sigmoid + BCELoss)
    # We add pos_weight because beats are rare (mostly 0s, few 1s)
    # This tells the loss function: "Pay 10x more attention to the BEATS (1s)"
    pos_weight = torch.tensor([10.0] * 4).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(tokens) # Shape: (1, Time, 4)
        
        # Loss
        loss = criterion(predictions, labels)
        
        # Background
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save Model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "beatmap_lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to .pt dataset file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    
    train(args.dataset_path, args.epochs)
