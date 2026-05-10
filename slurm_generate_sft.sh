#!/bin/bash
#SBATCH --job-name=sft_generate
#SBATCH --output=logs/sft_generate_%j.out
#SBATCH --error=logs/sft_generate_%j.err
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

echo "=============================================="
echo "  SFT Batch Generation (20 Songs, All Models)"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

mkdir -p logs
mkdir -p outputs

PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

echo "Fixing MuMu missing dependency (nnAudio)..."
/data/mg546924/conda_envs/deepresonance_env/bin/python -m pip install nnAudio --user

echo "Starting Batch Generation..."
$PYTHON scripts/batch_generate_sft_all_models.py --max-songs 20

echo "=============================================="
echo "[OK] Generation Complete!"
echo "   End : $(date)"
echo "   Now you can run slurm_benchmark_sft.sh to evaluate them!"
echo "=============================================="
