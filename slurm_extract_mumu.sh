#!/bin/bash
#SBATCH --job-name=extract_mumu
#SBATCH --output=logs/extract_mumu_%j.out
#SBATCH --error=logs/extract_mumu_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting MuMu Onset Extraction..."
mkdir -p logs

PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"
$PYTHON scripts/extract_onsets_mumu.py

echo "MuMu Onset Extraction complete."
