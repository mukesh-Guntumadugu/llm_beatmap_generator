# Qwen Audio Probing Tool (`probe_qwen.py`)

## What is this tool?
Neural Network audio embeddings are essentially unreadable arrays of floating point numbers (the LLM "Black Box"). The `probe_qwen.py` script exists to mathematically prove what features Qwen *actually* understands and cares about when listening to an audio file, specifically for rhythmic games.

This script intercepts the Whisper-based `audio_tower` inside Qwen-Audio before the LLM turns the features into text. It mathematically crushes those hidden dimensions using **PCA (Principal Component Analysis)** into a single, time-based activation line.

By graphing this AI "Activation Spike" over the exact same timeline as a rigid `librosa` mathematical extraction of the song's **Onsets** and **BPM**, you can physically see if Qwen's internal spikes natively align with the drums on a graph.

## Files
1. **`src/probe_qwen.py`**: The main PyTorch/Librosa extraction Python script.
2. **`slurm_run_probe_qwen.sh`**: The Slurm batched job array that iterates across the `Fraxtil's Arrow Arrangements` music directory running the visualizer.

## How it Works
1. **The Ground Truth Curve**: Extracts a rigid mathematical Onset/BPM curve from the `.ogg` file using Librosa.
2. **The Output Target**: Skips the text-processor and feeds the `.ogg` directly into Qwen's internal embedding structure.
3. **The Math**: We `try/except` extract the deepest `hidden_states` layers from the Neural Encoders.
4. **The Visual**: Maps both curves into high-resolution Matplotlib PNG charts so you can see the correlation.

## How to Run (On HPC Cluster)

To process the entire Fraxtil directory, simply submit the Slurm batch file from the root directory of your project:
```bash
sbatch slurm_run_probe_qwen.sh
```

**Results:**
The batch job will create a new directory named `results_qwen_probe_fraxtil/` at the root of your project. After the batch completes (roughly ~3-5 minutes per song), that folder will be populated with PNG plots that you can download directly to your local Macbook and inspect!
