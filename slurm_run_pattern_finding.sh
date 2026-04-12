#!/bin/bash
#SBATCH --job-name=pattern_finding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/pattern_finding_%j.out

echo "=============================================="
echo "  Pattern Finding Pipeline"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  Start:  $(date)"
echo "=============================================="

cd /data/mg546924/llm_beatmap_generator

# Ensure local imports work correctly
export PYTHONPATH="${PYTHONPATH}:/data/mg546924/llm_beatmap_generator/src"

# Using deepresonance_env (assuming it has librosa, torch, etc.)
PYTHONUNBUFFERED=1 /data/mg546924/conda_envs/deepresonance_env/bin/python -u pattern_finding_approach/audio_feature_extraction.py

echo ""
echo "Finished: $(date)"
