#!/bin/bash
#SBATCH --job-name=test_flamingo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_flamingo_%j.out

echo "=== MUSIC-FLAMINGO TEST — Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export HF_HOME=/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH
/data/mg546924/music_flamingo_env/bin/python -u scripts/test_bk_flamingo.py
echo "=== DONE $(date) ==="
