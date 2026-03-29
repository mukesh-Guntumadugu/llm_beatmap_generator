#!/bin/bash
#SBATCH --job-name=benchmark_bpm
#SBATCH --output=logs/benchmark_bpm_%j.log
#SBATCH --error=logs/benchmark_bpm_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodelist=node001

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "Benchmarking Librosa BPM vs DeepResonance BPM"
echo "Job ID      : $SLURM_JOB_ID"
echo "Start       : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1

# Running using the Qwen/DeepResonance environment
/data/mg546924/conda_envs/qwenenv/bin/python scripts/benchmark_deepresonance_bpm.py

echo "✅ Evaluation Complete: $(date)"
