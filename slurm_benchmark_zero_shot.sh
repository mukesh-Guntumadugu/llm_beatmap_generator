#!/bin/bash
#SBATCH --job-name=benchmark_zero_shot
#SBATCH --output=logs/benchmark_zero_shot_%j.out
#SBATCH --error=logs/benchmark_zero_shot_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting Zero-Shot Benchmarking for all open-source models..."
date

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p outputs

# Use the absolute path to the Python binary in the qwenenv environment
PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

# Execute the zero-shot benchmarking script
$PYTHON scripts/benchmark_zero_shot_all_models.py --max-songs 20

echo "Benchmarking complete."
date
