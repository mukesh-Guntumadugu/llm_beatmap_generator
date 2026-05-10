#!/bin/bash
#SBATCH --job-name=zs_qwen
#SBATCH --output=logs/zs_qwen_%j.out
#SBATCH --error=logs/zs_qwen_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting Zero-Shot Benchmarking for Qwen..."
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
/data/mg546924/conda_envs/qwenenv/bin/python scripts/benchmark_zero_shot_qwen.py
echo "Benchmarking complete."
