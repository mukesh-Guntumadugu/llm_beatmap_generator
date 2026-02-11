import torch
import torch.nn as nn
import sys
import os
import argparse
import glob
import datetime

# --- LOGGER ---
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Add src to path to import train
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train import BeatmapLSTM, BeatmapDataset

class ManualLSTMCell:
    def __init__(self, input_size, hidden_size, weight_ih, weight_hh, bias_ih, bias_hh):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # PyTorch LSTM weights are concatenated (Input, Forget, Cell, Output)
        # Shape: (4*hidden_size, input_size) for ih
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        
    def step(self, x, h_prev, c_prev):
        # x: (batch, input_size)
        # h_prev: (batch, hidden_size)
        # c_prev: (batch, hidden_size)
        
        # Linear projections - Separated for analysis
        # W_ih * x + b_ih
        gates_ih = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        # W_hh * h + b_hh
        gates_hh = torch.mm(h_prev, self.weight_hh.t()) + self.bias_hh
        
        gates = gates_ih + gates_hh
        
        # Chunk into 4 gates: i, f, g, o
        i_gate_pre, f_gate_pre, g_gate_pre, o_gate_pre = gates.chunk(4, 1)
        
        # Detailed Forget Gate Logic
        ih_chunk = gates_ih.chunk(4, 1)
        hh_chunk = gates_hh.chunk(4, 1)
        
        # Index 1 is Forget Gate
        f_input_contrib = ih_chunk[1] # Contribution from current input
        f_hidden_contrib = hh_chunk[1] # Contribution from past hidden state
        
        # Activations
        i_t = torch.sigmoid(i_gate_pre)
        f_t = torch.sigmoid(f_gate_pre)
        g_t = torch.tanh(g_gate_pre)
        o_t = torch.sigmoid(o_gate_pre)
        
        # Cell State Update (Long Term Memory)
        # c_t = f_t * c_{t-1} + i_t * g_t
        c_t = f_t * c_prev + i_t * g_t
        
        # Hidden State Update (Short Term Memory)
        # h_t = o_t * tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        
        # We return gates for analysis
        return h_t, c_t, f_t, i_t, o_t, f_input_contrib, f_hidden_contrib

class ExplainableLSTM(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Copy embedding layers
        self.embeddings = original_model.embeddings
        self.input_proj = original_model.input_proj
        self.fc_lanes = original_model.fc_lanes
        
        # Extract LSTM weights for manual execution
        # Model has 2 layers.
        # Layer 0
        self.l0 = ManualLSTMCell(
            input_size=original_model.lstm.input_size,
            hidden_size=original_model.lstm.hidden_size,
            weight_ih=original_model.lstm.weight_ih_l0,
            weight_hh=original_model.lstm.weight_hh_l0,
            bias_ih=original_model.lstm.bias_ih_l0,
            bias_hh=original_model.lstm.bias_hh_l0
        )
        # Layer 1
        self.l1 = ManualLSTMCell(
            input_size=original_model.lstm.hidden_size, # input to L1 is output of L0
            hidden_size=original_model.lstm.hidden_size,
            weight_ih=original_model.lstm.weight_ih_l1,
            weight_hh=original_model.lstm.weight_hh_l1,
            bias_ih=original_model.lstm.bias_ih_l1,
            bias_hh=original_model.lstm.bias_hh_l1
        )
        
        self.hidden_dim = original_model.lstm.hidden_size
        
        # Store weights for inspection
        self.weights = {
            "L0_ih": original_model.lstm.weight_ih_l0,
            "L0_hh": original_model.lstm.weight_hh_l0,
            "L1_ih": original_model.lstm.weight_ih_l1,
            "L1_hh": original_model.lstm.weight_hh_l1
        }
        
    def forward_analyze(self, x):
        # x: (Batch, SeqLen, 32 codebooks)
        batch_size, seq_len, _ = x.shape
        
        # Embed and Sum
        emb_sum = 0
        for i, embedding in enumerate(self.embeddings):
            emb_sum = emb_sum + embedding(x[:, :, i])
        
        # Project
        x_in = self.input_proj(emb_sum) # (Batch, SeqLen, Hidden)
        
        # Init States
        h0_l0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c0_l0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        h0_l1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c0_l1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        h_prev_l0, c_prev_l0 = h0_l0, c0_l0
        h_prev_l1, c_prev_l1 = h0_l1, c0_l1
        
        # Storage for analysis
        history = []
        
        print(f"\n{'='*20} LSTM TIMESTEP LOOP {'='*20}")
        print(f"Looping from t=0 to {seq_len-1}...")
        
        for t in range(seq_len):
            # Input at time t
            print(f"\n--- Time Step t={t} ---")
            
            # 1. Raw Tokens
            raw_tokens_at_t = x[:, t, :] # (Batch, 32)
            print(f"1. RAW INPUT TOKENS (What you sent): {raw_tokens_at_t.tolist()}")
            
            # 2. Embedding process (Already done before loop in 'x_in', but conceptually happens here)
            x_t = x_in[:, t, :] # (Batch, Hidden)
            print(f"2. PROCESSED VECTOR (What LSTM sees): Shape {x_t.shape}")
            print(f"   FULL VALUES: {x_t[0].tolist()}")
            
            # 3. Code Location
            print("   -> Entering LSTM Cell (self.l0.forward)...")
            
            # Layer 0
            h_l0, c_l0, f_l0, i_l0, o_l0, f_in_l0, f_hid_l0 = self.l0.step(x_t, h_prev_l0, c_prev_l0)
            
            # Layer 1 (Input is h_l0)
            h_l1, c_l1, f_l1, i_l1, o_l1, f_in_l1, f_hid_l1 = self.l1.step(h_l0, h_prev_l1, c_prev_l1)
            
            # Store interesting stats
            # We'll focus on Layer 1 (the top layer) for "Concept" memory
            stats = {
                "t": t,
                "L1_forget_mean": f_l1.mean().item(),
                "L1_cell_mean": c_l1.mean().item(),
                "L1_hidden_mean": h_l1.mean().item(), # Short Term
                "L1_f_input_contrib_mean": f_in_l1.mean().item(),
                "L1_f_hidden_contrib_mean": f_hid_l1.mean().item(),
                "L1_forget_gate": f_l1.detach(),
            }
            history.append(stats)
            
            # Update prev
            h_prev_l0, c_prev_l0 = h_l0, c_l0
            h_prev_l1, c_prev_l1 = h_l1, c_l1
            
        return history, self.weights

def ascii_heatmap(tensor, width=20):
    # tensor: 1D tensor of values between 0 and 1
    # We want to visualize the vector.
    # To fit in terminal, we might only show first N units or use a condensed block.
    # Let's show first 16 units as a bar
    symbols = " .:-=+*#%@"
    res = ""
    for val in tensor[:width]:
        idx = int(val * (len(symbols) - 1))
        idx = max(0, min(idx, len(symbols)-1))
        res += symbols[idx]
    return f"[{res}]"

def analyze_memory():
    device = 'cpu'
    
    # 1. Load Model
    print("Loading model...")
    model = BeatmapLSTM()
    model_path = "src/training/beatmap_lstm.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Random Comparison Model
    print("Creating Random Model for comparison...")
    random_model = BeatmapLSTM()
    # No load state dict, so it's random
    random_model.to(device)
    random_model.eval()
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(model)
    print(f"\nThis confirms we are using an LSTM with {model.lstm.num_layers} layers and hidden dim {model.lstm.hidden_size}.")
    
    print("\ncreating Explainable LSTM...")
    explainer = ExplainableLSTM(model)
    explainer.to(device)
    explainer.eval()
    
    # 3. Load Sample Data
    # Detect model dimensions
    # 2. Random Comparison Model
    print("Creating Random Model for comparison...")
    random_model = BeatmapLSTM()
    random_model.to(device)
    random_model.eval()
    
    # 3. Load Data
    num_model_codebooks = len(model.embeddings)
    selected_ds = None
    
    # Parse Args manually inside function or assume passed?
    # Better to parse at top level, but let's check sys.argv here for simplicity or use argparse in main
    
    if len(sys.argv) > 1:
        user_path = sys.argv[1]
        print(f"User provided file: {user_path}")
        if os.path.exists(user_path):
             # Try to load
             try:
                # If it's a token file (only inputs), we need a customized Dataset or just load tensor directly
                # Because BeatmapDataset expects 'tokens' and 'targets' usually. 
                # Let's see if we can support raw token files too.
                data = torch.load(user_path)
                
                # Check format
                if isinstance(data, torch.Tensor):
                    # Likely just tokens tensor (Frames, 32) or (32, Frames)
                    print("File contains raw Tensor.")
                    if data.dim() == 2:
                        if data.shape[1] == num_model_codebooks:
                             # (Frames, 32)
                             pass 
                        elif data.shape[0] == num_model_codebooks:
                             # (32, Frames) -> Transpose
                             data = data.t()
                        
                        # Create dummy dataset-like object
                        # We need x, y. y will be dummy.
                        x = data
                        y = torch.zeros(x.shape[0], 4) # Dummy targets
                        selected_ds = [(x, y)] # List of 1 sample
                        print("\n" + "!"*60)
                        print("WARNING: Loaded RAW TOKENS only (Inference Input).")
                        print("We do NOT have the true beatmap labels for this file.")
                        print("The 'Target' column in the output will be all '0000' (Dummy).")
                        print("!"*60 + "\n")
                    else:
                        print(f"Tensor shape {data.shape} not supported directly.")

                elif isinstance(data, dict):
                    # Standard Dataset dict
                     ds = BeatmapDataset(user_path, seq_len=32)
                     if len(ds) > 0:
                         selected_ds = ds
                         print("Loaded valid BeatmapDataset.")
                     else:
                        print("Dataset empty.")
             except Exception as e:
                 print(f"Error loading user file: {e}")
        else:
            print(f"File not found: {user_path}")

    # Fallback to Auto-Detect if nothing selected
    if selected_ds is None:
        print("Auto-detecting dataset...")
        data_pattern = "src/Neural Audio Codecs/outputs/*_dataset.pt"
        all_files = glob.glob(data_pattern)
        
        for f in all_files:
            try:
                ds = BeatmapDataset(f, seq_len=32)
                if len(ds) > 0 and ds.tokens.shape[1] == num_model_codebooks:
                    print(f"Found compatible dataset: {f}")
                    selected_ds = ds
                    break
            except Exception:
                pass

    if selected_ds is None:
        print("No compatible dataset found!")
        return
        
    # Get one sample
    x, y = selected_ds[0] # x: (SeqLen, 32), y: (SeqLen, 4)
    x_batch = x.unsqueeze(0).to(device) # Batch dim
    y_batch = y.unsqueeze(0).to(device)
    
    # Deep Dive Run
    history, weights = explainer.forward_analyze(x_batch)
    
    # Get Predictions
    with torch.no_grad():
        logits_trained = model(x_batch) # (1, Seq, 4, 6)
        logits_random = random_model(x_batch)
        
        # Get argmax for specific step prediction
        preds_trained = logits_trained.argmax(dim=3).squeeze(0) # (Seq, 4)
        preds_random = logits_random.argmax(dim=3).squeeze(0) # (Seq, 4)
    
    print("\n" + "="*80)
    print("GRANULAR INSPECTION: Step-by-Step")
    print("="*80)
    print(f"Showing first 16 neurons of the Forget Gate vector per step.")
    print(f"Heatmap Legend: ' ' (0.0) -> '.' -> ':' -> '*' -> '@' (1.0)")
    print("-" * 80)
    print(f"{'t':<3} | {'In(0)':<6} | {'Forget Gate Vector (First 20)':<24} | {'Target':<10} | {'Trained':<10} | {'Random':<10} | {'Status'}")
    print("-" * 80)
    
    correct_trained_count = 0
    correct_random_count = 0
    total_lanes = 0
    
    for step in history:
        t = step['t']
        f_vec = step['L1_forget_gate'][0] # Take first item in batch
        
        # Visualize Vector
        f_viz = ascii_heatmap(f_vec, width=20)
        
        # Input Token (First codebook only for brevity)
        in_token = x[t, 0].item()
        
        # Targets & Preds (All 4 lanes)
        # Just show Lane 0 for brevity? Or compact format: 1000
        target_lanes = y[t].tolist()
        pred_t_lanes = preds_trained[t].tolist()
        pred_r_lanes = preds_random[t].tolist()
        
        # Basic accuracy check for this step
        # Note: Class 0 is "Empty". We mainly care if we predict notes (Class > 0)
        # Let's count exact matches
        is_exact = (target_lanes == pred_t_lanes)
        status = "OK" if is_exact else "MISS"
        
        # Stats
        for i in range(4):
            if target_lanes[i] == pred_t_lanes[i]: correct_trained_count += 1
            if target_lanes[i] == pred_r_lanes[i]: correct_random_count += 1
            total_lanes += 1
            
        # Format strings
        tgt_str = "".join(str(i) for i in target_lanes)
        trn_str = "".join(str(i) for i in pred_t_lanes)
        rnd_str = "".join(str(i) for i in pred_r_lanes)
        
        print(f"{t:<3} | {in_token:<6} | {f_viz:<24} | {tgt_str:<10} | {trn_str:<10} | {rnd_str:<10} | {status}")

    print("\n" + "="*80)
    print("DOES TRAINING HELP?")
    print("="*80)
    trained_acc = (correct_trained_count / total_lanes) * 100
    random_acc = (correct_random_count / total_lanes) * 100
    
    print(f"Trained Model Accuracy: {trained_acc:.2f}%")
    print(f"Random Model Accuracy:  {random_acc:.2f}%")
    print(f"Difference:             +{trained_acc - random_acc:.2f}%")
    
    if trained_acc > random_acc + 10:
        print("\nCONCLUSION: Yes! The training is helping significantly.")
        print("The trained model accurately predicts patterns that the random model misses.")
        print("The Forget Gate Visualization shows structured memory retention (dots and stars).")
    else:
        print("\nCONCLUSION: The model might be underfitting or the sequence is mostly empty (Class 0).")
        print("Check if the Target is mostly '0000'. If so, high accuracy might just mean 'predicting 0 everywhere'.")

if __name__ == "__main__":
    # Setup Logging
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"analysis_log_{timestamp}.log")
    
    # Redirect stdout to TeeLogger
    sys.stdout = TeeLogger(log_filename)
    
    # Enable Full Printing
    torch.set_printoptions(threshold=sys.maxsize, linewidth=200, sci_mode=False)
    
    print(f"Logging analysis to {log_filename}...")
    
    with torch.no_grad():
        analyze_memory()
        
    print(f"\nAnalysis log saved to {log_filename}")
