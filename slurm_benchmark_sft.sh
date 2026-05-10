#!/bin/bash
#SBATCH --job-name=benchmark_sft
#SBATCH --output=logs/benchmark_sft_%j.out
#SBATCH --error=logs/benchmark_sft_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4

echo "Starting SFT Benchmarking from generated .ssc files..."
date

mkdir -p logs

# Use standard base python environment
source /data/mg546924/conda_envs/qwenenv/bin/activate

python scripts/benchmark_sft_from_ssc.py

echo "Benchmarking complete."
date
