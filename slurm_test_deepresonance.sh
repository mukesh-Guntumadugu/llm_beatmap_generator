#!/bin/bash
#SBATCH --job-name=test_deepres
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_deepresonance_%j.out

echo "=== DEEPRESONANCE TEST — Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH
/data/mg546924/conda_envs/deepresonance_env/bin/python -u scripts/test_bk_deepresonance.py
echo "=== DONE $(date) ==="
