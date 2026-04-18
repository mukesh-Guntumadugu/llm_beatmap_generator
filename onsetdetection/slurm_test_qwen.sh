#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_qwen_%j.out

echo "=== QWEN TEST — Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH
/data/mg546924/conda_envs/qwenenv/bin/python -u scripts/test_bk_qwen.py

echo "=== SCORING QWEN OUTPUT ==="
/data/mg546924/conda_envs/deepresonance_env/bin/python /data/mg546924/llm_beatmap_generator/scripts/compare_onsets.py --tolerance 50

echo "=== DONE $(date) ==="
