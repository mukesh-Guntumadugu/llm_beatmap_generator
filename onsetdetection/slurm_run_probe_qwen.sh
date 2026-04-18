#!/bin/bash
#SBATCH --job-name=qwen_probe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_probe_%j.out
#SBATCH --error=logs/qwen_probe_%j.err

echo "=== QWEN AUDIO PROBING SCRIPT === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1

# Load Qwen Conda environment
PYTHON_DRIVER="/data/mg546924/conda_envs/qwenenv/bin/python"
TARGET_DIR="src/musicForBeatmap/Fraxtil's Arrow Arrangements"
OUTPUT_DIR="results_qwen_probe_fraxtil"

echo "--------------------------------------------------------"
echo "Targeting folder: $TARGET_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "--------------------------------------------------------"

$PYTHON_DRIVER -u src/probe_qwen.py --target_dir "$TARGET_DIR" --output_dir "$OUTPUT_DIR"

echo "=== PROBING COMPLETE === $(date)"
