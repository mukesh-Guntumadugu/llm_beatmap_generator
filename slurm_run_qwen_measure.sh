#!/bin/bash
#SBATCH --job-name=qwen_measure
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen_measure_%j.out

echo "=== QWEN MEASURE-BY-MEASURE: Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export BENCHMARK_AUDIO="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"
export BENCHMARK_OUT="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup"

# Fix dynamic link paths if needed
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

echo "Running target script..."
/data/mg546924/conda_envs/qwenenv/bin/python -u scripts/Qwen/qwen_measure_generator.py

echo "=== DONE $(date) ==="
