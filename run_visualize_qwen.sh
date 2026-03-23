#!/bin/bash
#SBATCH --job-name=qwen_viz
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen_viz_%j.out
#SBATCH --error=logs/qwen_viz_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Using conda env: /data/mg546924/conda_envs/qwenenv/bin"

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# Fix Numpy / Matplotlib conflict before running
echo "Fixing Numpy compatibility..."
/data/mg546924/conda_envs/qwenenv/bin/python3 -m pip install "numpy<2"

# Run the visualization script
echo "Extracting Qwen Internal Spectrogram..."
/data/mg546924/conda_envs/qwenenv/bin/python3 visualize_qwen_internal.py

echo "Job ended: $(date)"
