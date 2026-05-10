#!/bin/bash
#SBATCH --job-name=zs_flam
#SBATCH --output=logs/zs_flamingo_%j.out
#SBATCH --error=logs/zs_flamingo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Starting Zero-Shot Benchmarking for Flamingo..."
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
/data/mg546924/conda_envs/flamingo_env/bin/python scripts/benchmark_zero_shot_flamingo.py
echo "Benchmarking complete."
