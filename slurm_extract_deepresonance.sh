#!/bin/bash
#SBATCH --job-name=extract_deepresonance
#SBATCH --output=logs/extract_deepresonance_%j.out
#SBATCH --error=logs/extract_deepresonance_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting DeepResonance Onset Extraction..."
mkdir -p logs

PYTHON="/data/mg546924/conda_envs/deepresonance_env/bin/python"
$PYTHON scripts/extract_onsets_deepresonance.py

echo "DeepResonance Onset Extraction complete."
