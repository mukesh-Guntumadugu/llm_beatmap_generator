#!/bin/bash
#SBATCH --job-name=mumu_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=mumu_log_%j.txt

echo "=== Starting MuMu-LLaMA Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="
cd /data/mg546924/llm_beatmap_generator

# Redirect user site-packages to /data (home quota is full)
export PYTHONUSERBASE=/data/mg546924/.local
export PATH=$PYTHONUSERBASE/bin:$PATH

# Use the conda env's Python directly (most reliable in SLURM batch scripts)
/home/mg546924/.conda/envs/mumullama/bin/python onsetdetection/extract_mumu_onsets.py

echo "=== MuMu-LLaMA Onset Detection Finished ==="
