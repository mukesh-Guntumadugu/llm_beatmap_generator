import torch
import torch.nn as nn
import sys
import os
import logging
import random

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.train import BeatmapLSTM, SpringtimeDataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def trace_lstm_step(model, input_token_idx, h_prev, c_prev):
    """
    Manually computes the gates for a single LSTM step (Layer 0).
    """
    # 1. Embedding
    # input_token_idx is a scalar tensor for one token
    embedded = model.embedding(input_token_idx.unsqueeze(0)) # (1, embed_dim)
    x = embedded.squeeze(0) # (embed_dim)
    
    # 2. LSTM Weights (Layer 0)
    # Weights are stored as (4*hidden_dim, input_dim)
    # PyTorch order: Input, Forget, Cell, Output (i, f, g, o)
    w_ih = model.lstm.weight_ih_l0
    w_hh = model.lstm.weight_hh_l0
    b_ih = model.lstm.bias_ih_l0
    b_hh = model.lstm.bias_hh_l0
    
    hidden_dim = model.lstm.hidden_size
    
    # Linear transformations
    gates_ih = torch.matmul(x, w_ih.t()) + b_ih
    gates_hh = torch.matmul(h_prev, w_hh.t()) + b_hh
    gates = gates_ih + gates_hh # (4*hidden_dim)
    
    # Split into 4 gates
    chunk_size = hidden_dim
    i_gate_pre = gates[0:chunk_size]
    f_gate_pre = gates[chunk_size:2*chunk_size]
    g_gate_pre = gates[2*chunk_size:3*chunk_size]
    o_gate_pre = gates[3*chunk_size:4*chunk_size]
    
    # Activations
    i_gate = sigmoid(i_gate_pre)
    f_gate = sigmoid(f_gate_pre)
    g_gate = torch.tanh(g_gate_pre) # Cell candidate uses Tanh
    o_gate = sigmoid(o_gate_pre)
    
    # 3. Cell State Update
    c_new = (f_gate * c_prev) + (i_gate * g_gate)
    
    # 4. Hidden State Update
    h_new = o_gate * torch.tanh(c_new)
    
    print("\n--- LSTM Computation Trace (Layer 0, Step 1) ---")
    print(f"Input Token Index: {input_token_idx.item()}")
    print(f"Embedding Vector (Input to LSTM) [Shape {x.shape}]:")
    print(x)
    
    print(f"\n[Forget Gate] Controls what to keep from previous memory.")
    print(f"  Pre-activation (W_f*x + U_f*h + b):")
    print(f_gate_pre)
    print(f"  Activation (Sigmoid):")
    print(f_gate)

    print(f"\n[Input Gate] Controls what new info to add.")
    print(f"  Activation (Sigmoid):")
    print(i_gate)
    
    print(f"\n[Cell Candidate] The new information proposal.")
    print(f"  Activation (Tanh):")
    print(g_gate)
    
    print(f"\n[Cell State Update] c_new = f * c_prev + i * g")
    print(f"  Previous Cell State:")
    print(c_prev)
    print(f"  New Cell State:")
    print(c_new)
    
    print(f"\n[Output Gate] Controls what part of cell state to output.")
    print(f"  Activation (Sigmoid):")
    print(o_gate)
    
    print(f"\n[Hidden State Output] h_new = o * tanh(c_new)")
    print(f"  New Hidden State:")
    print(h_new)
    
    return h_new, c_new

def main():
    # Load Model
    device = 'cpu'
    model = BeatmapLSTM()
    model_path = "src/training/beatmap_lstm_springtime_best.pth"
    
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            return
    else:
        print(f"Model file not found at {model_path}. Using random weights.")
        
    model.eval()
    
    # Load Sample Data
    # Assuming user still has the single file setup in train.py or the file exists
    # We'll just define the path manually to match what we know exists
    token_file = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv"
    ssc_file = "src/musicForBeatmap/Springtime/beatmap_easy.text"
    
    # Configure full printing
    torch.set_printoptions(profile="full", linewidth=200)

    try:
        ds = SpringtimeDataset(token_csv_path=token_file, ssc_path=ssc_file)
        if len(ds) == 0:
            print("Dataset empty.")
            return
            
        # Get one sample
        # x shape: (1200,)
        x_sample, _, _ = ds[0] 
        print(f"\n=== INPUT DATA (FULL SEQUENCE) ===")
        print(f"Total Sequence Length: {x_sample.shape[0]} Integers.")
        print(f"These are the EnCodec tokens feeding into the LSTM:")
        print(x_sample)
        
        # We'll trace the FIRST token processing
        first_token = x_sample[0]
        
        # Initialize hidden state
        hidden_dim = model.lstm.hidden_size
        h_prev = torch.zeros(hidden_dim)
        c_prev = torch.zeros(hidden_dim)
        
        # Run trace
        h_new, c_new = trace_lstm_step(model, first_token, h_prev, c_prev)
        
        # Show higher level prediction
        print(f"\n--- Final Prediction Mechanism ---")
        with torch.no_grad():
            x_batch = x_sample.unsqueeze(0) # (1, 1200)
            notes_out, density_out = model(x_batch)
            
            # Take the last step output as example
            last_step_notes = notes_out[0]
            
            print(f"Output of LSTM passes through Linear Layers:")
            print(f"1. `fc_notes`: Predicts notes for 16 time slots (1/16th notes).")
            print(f"   Shape: {last_step_notes.shape}")
            print(f"   Full Logits for Slot 0, Column 0:")
            print(last_step_notes[0, 0])
             
            print(f"2. `fc_density`: Predicts note density class.")
            print(f"   Full Logits:")
            print(density_out[0])
            print(f"   Predicted Class: {torch.argmax(density_out[0]).item()}")

    except Exception as e:
        print(f"Error running trace: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
