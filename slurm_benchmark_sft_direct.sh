#!/bin/bash
#SBATCH --job-name=benchmark_sft_direct
#SBATCH --output=logs/benchmark_sft_direct_%j.out
#SBATCH --error=logs/benchmark_sft_direct_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting Direct SFT Onset Benchmarking (Raw Milliseconds)..."
date

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p outputs

# Use the absolute path to the Python binary in the qwenenv environment
PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

# Ensure librosa is installed
$PYTHON -m pip install librosa soundfile --user

# Execute the direct onset benchmarking script
$PYTHON scripts/benchmark_sft_onsets_direct.py --max-songs 20

echo "Benchmarking complete."
date
