#!/bin/bash
#SBATCH --job-name=train_flamingo
#SBATCH --output=logs/train_flamingo_%j.log
#SBATCH --error=logs/train_flamingo_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "Training Music-Flamingo (5 Epochs | 5s Chunks)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Start       : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))
export DS_SKIP_CUDA_CHECK=1
# Force HuggingFace to use only locally cached files (no internet on compute nodes)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

/data/mg546924/conda_envs/qwenenv/bin/python3.9 scripts/train_flamingo_5s_lora.py

echo "✅ Music-Flamingo Training Complete: $(date)"
