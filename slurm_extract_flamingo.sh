#!/bin/bash
#SBATCH --job-name=extract_flamingo
#SBATCH --output=logs/extract_flamingo_%j.out
#SBATCH --error=logs/extract_flamingo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "Starting Flamingo Onset Extraction..."
mkdir -p logs

PYTHON="/data/mg546924/music_flamingo_env/bin/python"
$PYTHON scripts/extract_onsets_flamingo.py

echo "Flamingo Onset Extraction complete."
