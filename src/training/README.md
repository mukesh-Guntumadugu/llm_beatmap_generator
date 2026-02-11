# Training Directory

This directory contains all training scripts for the Beatmap LSTM model.

## Files

### Core Training
- **`train.py`** - Original training script with local logging and PNG graph output
- **`train_with_wandb.py`** - Enhanced training with Weights & Biases visualization ⭐ **NEW!**
- **`quick_start_wandb.py`** - Interactive setup wizard for W&B training ⭐ **NEW!**

### Analysis Tools
- **`analyze_memory.py`** - Analyze LSTM internal states
- **`inspect_training_data.py`** - Validate training data format
- **`dump_pt_to_csv.py`** - Export .pt files to CSV
- **`explain_pt_file.py`** - Display .pt file contents

### Directories
- **`logs/`** - Training logs
- **`models/`** - Saved model checkpoints

## Quick Start

### Option 1: W&B Training (Recommended - Beautiful Graphs!)
```bash
python quick_start_wandb.py
```
This will guide you through setup and start training with interactive W&B visualizations.

### Option 2: Traditional Training
```bash
python train.py
```
Uses the original method with local PNG graphs.

## What's New with W&B?

The new `train_with_wandb.py` provides:
- 📊 **Interactive dashboards** instead of static PNG files
- 🌐 **Cloud-based** - access from anywhere
- 🔄 **Real-time updates** - watch training live
- 📈 **Advanced metrics** - confusion matrices, per-direction analytics
- 🔗 **Shareable** - send a URL instead of files
- 🆓 **Free** for personal use

See [WANDB_TRAINING.md](../../docs/WANDB_TRAINING.md) for complete documentation.

## Training Flow

1. **Prepare data**: Ensure you have token CSV and beatmap text files
2. **Choose method**: W&B (new) or traditional (existing)
3. **Run training**: Follow prompts or use defaults
4. **Monitor progress**: W&B dashboard or local logs
5. **Use model**: Load best checkpoint for inference

## Model Checkpoints

Training saves models as:
- `beatmap_lstm_{song_name}_best.pth` - Best performing epoch
- `beatmap_lstm_{song_name}_final.pth` - Final epoch (W&B only)

## Requirements

For W&B training, install:
```bash
pip install wandb
wandb login
```

See `../../requirements_wandb.txt` for version info.

## Troubleshooting

**W&B not installed?**
```bash
pip install wandb
```

**Not logged in?**
```bash
wandb login
```

**Using CPU instead of GPU/MPS?**
Check your PyTorch installation and device availability.

For more help, see the [documentation](../../docs/WANDB_TRAINING.md).
