#!/usr/bin/env python3
"""
Quick Start Script for W&B Training
====================================
This script provides a simple, interactive way to start training with W&B visualization.
Just run this script and it will guide you through the setup!
"""

import os
import sys

def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb
        return True
    except ImportError:
        return False

def install_wandb():
    """Install wandb package."""
    print("📦 Installing Weights & Biases...")
    os.system("pip install wandb")
    print("✅ Installation complete!")

def login_wandb():
    """Login to wandb."""
    import wandb
    print("\n🔐 Logging in to Weights & Biases...")
    print("This will open a browser window to get your API key.")
    wandb.login()
    print("✅ Login successful!")

def main():
    print("="*60)
    print("🎵 Beatmap LSTM Training with Weights & Biases")
    print("="*60)
    print()
    
    # Step 1: Check if wandb is installed
    if not check_wandb_installed():
        print("❌ Weights & Biases is not installed.")
        response = input("Would you like to install it now? (y/n): ").lower().strip()
        if response == 'y':
            install_wandb()
        else:
            print("❌ Cannot proceed without wandb. Exiting.")
            sys.exit(1)
    else:
        print("✅ Weights & Biases is installed")
    
    # Step 2: Login to wandb
    import wandb
    if not wandb.api.api_key:
        print("\n❌ You are not logged in to Weights & Biases.")
        response = input("Would you like to login now? (y/n): ").lower().strip()
        if response == 'y':
            login_wandb()
        else:
            print("❌ Cannot proceed without login. Exiting.")
            sys.exit(1)
    else:
        print("✅ Already logged in to Weights & Biases")
    
    # Step 3: Import training modules
    print("\n📚 Loading training modules...")
    try:
        from train_with_wandb import train_with_wandb
        from train import SpringtimeDataset, BeatmapLSTM, setup_logging
        from torch.utils.data import DataLoader
        import torch
        print("✅ Modules loaded successfully")
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        sys.exit(1)
    
    # Step 4: Setup paths
    print("\n📁 Setting up file paths...")
    token_file = "src/tokens_generated/generate_chart_Encodex_tokens_Kommisar - Springtime_20260127231509.csv"
    ssc_file = "src/musicForBeatmap/Springtime/beatmap_easy.text"
    
    # Extract song name
    token_basename = os.path.basename(token_file)
    if "tokens_" in token_basename:
        song_name = token_basename.split("tokens_")[1].split("_202")[0]
    else:
        song_name = "Springtime"
    
    print(f"   Song: {song_name}")
    print(f"   Tokens: {token_file}")
    print(f"   Beatmap: {ssc_file}")
    
    # Step 5: Create dataset
    print("\n🗂️  Creating dataset...")
    try:
        ds = SpringtimeDataset(token_csv_path=token_file, ssc_path=ssc_file)
        print(f"✅ Dataset created with {len(ds)} samples")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        sys.exit(1)
    
    # Step 6: Create dataloader
    print("\n📦 Creating dataloader...")
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    print("✅ Dataloader ready")
    
    # Step 7: Create model
    print("\n🤖 Creating LSTM model...")
    model = BeatmapLSTM()
    print("✅ Model created")
    
    # Step 8: Check for existing weights
    best_weights_path = f"src/training/beatmap_lstm_{song_name}_best.pth"
    if os.path.exists(best_weights_path):
        print(f"\n💾 Found existing weights: {best_weights_path}")
        response = input("Load existing weights? (y/n): ").lower().strip()
        if response == 'y':
            model.load_state_dict(torch.load(best_weights_path, weights_only=True))
            print("✅ Weights loaded")
        else:
            print("🆕 Starting from scratch")
    else:
        print("\n🆕 No existing weights found - starting from scratch")
    
    # Step 9: Select device
    print("\n💻 Detecting compute device...")
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (this will be slower)")
    
    # Step 10: Get training parameters
    print("\n⚙️  Training configuration:")
    try:
        epochs = int(input("Number of epochs (default 300): ") or "300")
        learning_rate = float(input("Learning rate (default 0.001): ") or "0.001")
        project_name = input("W&B Project name (default 'beatmap-lstm'): ") or "beatmap-lstm"
    except ValueError:
        print("❌ Invalid input. Using defaults.")
        epochs = 300
        learning_rate = 0.001
        project_name = "beatmap-lstm"
    
    print(f"\n   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Project: {project_name}")
    print(f"   Device: {device}")
    
    # Step 11: Confirm and start training
    print("\n" + "="*60)
    print("🚀 Ready to start training!")
    print("="*60)
    response = input("\nStart training now? (y/n): ").lower().strip()
    
    if response != 'y':
        print("❌ Training cancelled.")
        sys.exit(0)
    
    # Step 12: Start training!
    print("\n" + "="*60)
    print("🎯 STARTING TRAINING WITH WEIGHTS & BIASES")
    print("="*60)
    print()
    
    setup_logging()
    
    try:
        trained_model = train_with_wandb(
            model=model,
            dataloader=loader,
            epochs=epochs,
            device=device,
            song_name=song_name,
            project_name=project_name,
            learning_rate=learning_rate,
            save_dir='src/training',
            log_interval=10
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 View your results at your W&B dashboard")
        print(f"💾 Model saved to: src/training/beatmap_lstm_{song_name}_best.pth")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(0)
