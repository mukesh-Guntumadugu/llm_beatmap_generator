#!/bin/bash
#SBATCH --job-name=build_5s_data
#SBATCH --output=logs/build_data_%j.log
#SBATCH --error=logs/build_data_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "Building 5-Second SFT Dataset on HPC Nodes"
echo "Job ID      : $SLURM_JOB_ID"
echo "Start       : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1

/data/mg546924/conda_envs/qwenenv/bin/python scripts/prepare_5s_sft_dataset.py

echo "✅ Dataset Construction Complete: $(date)"
