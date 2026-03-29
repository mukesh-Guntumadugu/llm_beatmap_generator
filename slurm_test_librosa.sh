#!/bin/bash
#SBATCH --job-name=test_librosa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_librosa_%j.out

echo "=== LIBROSA TEST — Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
/data/mg546924/conda_envs/deepresonance_env/bin/python -u scripts/test_bk_librosa.py
echo "=== DONE $(date) ==="
