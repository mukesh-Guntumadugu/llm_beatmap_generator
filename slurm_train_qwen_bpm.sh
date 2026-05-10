#!/bin/bash
#SBATCH --job-name=qwen_bpm_sft
#SBATCH --output=logs/qwen_bpm_sft_%j.log
#SBATCH --error=logs/qwen_bpm_sft_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator

# CRITICAL: Only use 1 GPU to prevent SLURM DataParallel deadlock
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  Qwen2-Audio Standalone BPM SFT Experiment"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

mkdir -p logs
mkdir -p outputs

PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

# 1. Generate Dataset
echo "Generating BPM JSONL Dataset..."
$PYTHON scripts/prepare_bpm_sft_dataset.py

# 2. Run LoRA Fine-Tuning
echo "Starting LoRA Fine-Tuning..."
$PYTHON scripts/train_qwen_bpm_lora.py

echo "=============================================="
echo "[OK] Qwen BPM SFT Complete!"
echo "   End : $(date)"
echo "   Check outputs/qwen-bpm-lora/ for checkpoints."
echo "=============================================="
