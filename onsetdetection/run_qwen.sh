#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=qwen_log_%j.txt

echo "=== Starting Qwen2-Audio Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================="

cd /data/mg546924/llm_beatmap_generator

# Block ~/.local packages (binary incompatible with our new data-based env)
export PYTHONNOUSERSITE=1

# Use the conda env on /data (home quota was full, env recreated there)
/data/mg546924/conda_envs/qwenenv/bin/python generate_qwen_beatmap.py

echo "=== Qwen Onset Detection Finished ==="
