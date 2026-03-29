#!/bin/bash
#SBATCH --job-name=all_models_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/all_models_test_%j.out

echo "=============================================="
echo "  ALL MODELS BENCHMARK — Bad Ketchup"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  Start:  $(date)"
echo "=============================================="

cd /data/mg546924/llm_beatmap_generator

# -u = unbuffered: prints appear live in the log instead of all at the end
PYTHONUNBUFFERED=1 /data/mg546924/conda_envs/deepresonance_env/bin/python -u test_all_models_bad_ketchup.py

echo ""
echo "Finished: $(date)"
