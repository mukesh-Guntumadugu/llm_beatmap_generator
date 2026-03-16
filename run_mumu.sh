#!/bin/bash
#SBATCH --job-name=mumu_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=mumu_log_%j.txt

echo "=== Starting MuMu-LLaMA Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="

# Force Python to ignore ~/.local site-packages (prevents shadowing conda env)
export PYTHONNOUSERSITE=1

# Use the conda env's Python directly (most reliable in SLURM batch scripts)
/home/mg546924/.conda/envs/mumullama/bin/python extract_mumu_onsets.py

echo "=== MuMu-LLaMA Onset Detection Finished ==="
