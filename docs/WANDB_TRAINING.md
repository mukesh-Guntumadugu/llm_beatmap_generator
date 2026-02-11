# Training with Weights & Biases (W&B)

This guide explains how to use the new W&B-integrated training function to get beautiful, interactive training visualizations.

## 🎯 Why Use W&B?

Weights & Biases provides:
- **Interactive Dashboards**: View all metrics in real-time with beautiful, interactive graphs
- **Experiment Tracking**: Compare multiple training runs side-by-side
- **Automatic Logging**: Metrics, hyperparameters, and model checkpoints tracked automatically
- **Shareable Results**: Easily share training results with others via URL
- **Advanced Visualizations**: Confusion matrices, per-direction metrics, and more

## 📦 Installation

1. Install Weights & Biases:
```bash
pip install wandb
```

Or use the requirements file:
```bash
pip install -r requirements_wandb.txt
```

2. Login to W&B (one-time setup):
```bash
wandb login
```

This will open a browser window where you can get your API key. Paste it into the terminal.

## 🚀 Quick Start

### Basic Usage

```python
from train_with_wandb import train_with_wandb
from train import SpringtimeDataset, BeatmapLSTM
from torch.utils.data import DataLoader
import torch

# Create dataset
ds = SpringtimeDataset(
    token_csv_path="src/tokens_generated/your_tokens.csv",
    ssc_path="src/musicForBeatmap/your_song/beatmap_easy.text"
)

# Create dataloader
loader = DataLoader(ds, batch_size=32, shuffle=False)

# Create model
model = BeatmapLSTM()

# Train with W&B visualization
train_with_wandb(
    model=model,
    dataloader=loader,
    epochs=300,
    device='mps',  # or 'cuda' or 'cpu'
    song_name='Springtime',
    project_name='beatmap-lstm-training'
)
```

### Running the Example Script

```bash
cd /Users/mukeshguntumadugu/llm_beatmap_generator
python src/training/train_with_wandb.py
```

## 📊 What Gets Tracked?

The W&B training function tracks everything the original training does, plus more:

### Loss Metrics
- Total loss (combined notes + density)
- Notes loss (separate)
- Density loss (separate)
- Perplexity (exp of loss)

### Accuracy Metrics
- Density accuracy (%)
- Perfect beatmap match rate (%)
- Note-level accuracy (%)
- F1-score (macro average)

### Per-Direction Metrics
For each arrow direction (Left, Down, Up, Right):
- Precision (%)
- Recall (%)
- F1-score (%)

### Additional Tracking
- Training time per epoch
- Confusion matrices (notes and density)
- Model gradients and weights
- Best model checkpoints
- Hyperparameters (learning rate, batch size, etc.)

## 🎨 Dashboard Features

When you run training with W&B, you'll get:

1. **Real-time Graphs**: Watch your metrics update live as training progresses
2. **Multiple Y-Axes**: Compare metrics with different scales on the same graph
3. **Smoothing**: Apply smoothing to noisy metrics for clearer trends
4. **Zoom & Pan**: Interactive graphs you can explore in detail
5. **Download**: Export graphs as images or data
6. **Share**: Get a URL to share your training results with others

## 🔧 Advanced Configuration

### Custom Project and Run Names

```python
train_with_wandb(
    model=model,
    dataloader=loader,
    epochs=300,
    device='mps',
    song_name='Springtime',
    project_name='my-beatmap-project',  # Custom project name
    run_name='experiment_v2'  # Custom run name
)
```

### Adjust Learning Rate

```python
train_with_wandb(
    model=model,
    dataloader=loader,
    learning_rate=5e-4,  # Custom learning rate
    # ... other params
)
```

### Change Logging Frequency

```python
train_with_wandb(
    model=model,
    dataloader=loader,
    log_interval=5,  # Log confusion matrices every 5 epochs instead of 10
    # ... other params
)
```

## 📁 File Organization

The W&B training saves models to:
- `src/training/beatmap_lstm_{song_name}_best.pth` - Best model during training
- `src/training/beatmap_lstm_{song_name}_final.pth` - Final model after all epochs

## 🆚 Comparison: Original vs W&B Training

| Feature | Original `train()` | W&B `train_with_wandb()` |
|---------|-------------------|--------------------------|
| Loss tracking | ✅ Local file | ✅ Cloud dashboard |
| Accuracy metrics | ✅ Local file | ✅ Cloud dashboard |
| Static graphs | ✅ PNG files | ✅ Interactive plots |
| Confusion matrices | ❌ Not logged | ✅ Interactive matrices |
| Per-direction metrics | ✅ Logged to file | ✅ Beautiful charts |
| Hyperparameter tracking | ❌ Manual | ✅ Automatic |
| Experiment comparison | ❌ Manual | ✅ Built-in |
| Sharing results | ❌ Send files | ✅ Send URL |
| Real-time monitoring | ❌ Check files | ✅ Live dashboard |

## 💡 Tips & Best Practices

1. **Name your runs meaningfully**: Use descriptive run names to identify experiments
2. **Use projects to organize**: Group related experiments in the same project
3. **Check the dashboard frequently**: Catch training issues early by monitoring live
4. **Compare runs**: Use W&B's comparison view to find the best hyperparameters
5. **Add notes**: Use W&B's notes feature to document what you changed between runs

## 🔍 Example Output

When you run training, you'll see:

```
2026-02-09 16:04:29,830 [INFO] ============================================================
2026-02-09 16:04:29,830 [INFO] TRAINING WITH WANDB (Project: beatmap-lstm-training, Run: Springtime_lstm_20260209_160429)
2026-02-09 16:04:29,830 [INFO] Input: 300 Frames (1200 Tokens) -> Output: 1s Beatmap
2026-02-09 16:04:29,830 [INFO] ============================================================
2026-02-09 16:04:29,830 [INFO] 🔗 View your training dashboard at: https://wandb.ai/your-username/beatmap-lstm-training/runs/...
```

Click the URL to open your interactive dashboard!

## 🐛 Troubleshooting

### "wandb not found"
```bash
pip install wandb
```

### "Not logged in to wandb"
```bash
wandb login
```

### "MPS device not available"
Change `device='mps'` to `device='cpu'`

## 📚 Learn More

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [Example Projects](https://wandb.ai/gallery)

---

**Happy Training! 🎵🎮**
