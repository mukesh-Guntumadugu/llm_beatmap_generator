#!/bin/bash
#SBATCH --job-name=zs_deep
#SBATCH --output=logs/zs_deepresonance_%j.out
#SBATCH --error=logs/zs_deepresonance_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting Zero-Shot Benchmarking for DeepResonance..."
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
/data/mg546924/conda_envs/deepresonance_env/bin/python scripts/benchmark_zero_shot_deepresonance.py
echo "Benchmarking complete."
