#!/bin/bash
#SBATCH --job-name=sft_gen_qwen
#SBATCH --output=logs/sft_gen_qwen_%j.out
#SBATCH --error=logs/sft_gen_qwen_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
mkdir -p outputs

PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

echo "Starting Qwen SFT Batch Generation..."
$PYTHON scripts/batch_generate_sft_all_models.py --max-songs 20 --model Qwen

echo "Qwen Generation Complete!"
