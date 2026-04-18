#!/bin/bash
#SBATCH --job-name=probe_comparison
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/probe_comparison_%j.out
#SBATCH --error=logs/probe_comparison_%j.err

echo "=== FLAMINGO & DEEPRESONANCE ARCHITECTURAL PROBING === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1
TARGET_DIR="src/musicForBeatmap/Fraxtil's Arrow Arrangements"

echo "--------------------------------------------------------"
echo "PHASE 1: PROBING FLAMINGO (AudioMAE Backbone)"
echo "--------------------------------------------------------"
PYTHON_DRIVER_FLAMINGO="/data/mg546924/conda_envs/music_flamingo_env/bin/python"
OUTPUT_DIR_FLAMINGO="results_flamingo_probe_fraxtil"

$PYTHON_DRIVER_FLAMINGO -u src/probe_flamingo.py --target_dir "$TARGET_DIR" --output_dir "$OUTPUT_DIR_FLAMINGO"

echo "--------------------------------------------------------"
echo "PHASE 2: PROBING DEEPRESONANCE (ImageBind Backbone)"
echo "--------------------------------------------------------"
PYTHON_DRIVER_DR="/data/mg546924/conda_envs/deepresonance_env/bin/python"
OUTPUT_DIR_DR="results_dr_probe_fraxtil"

# Mandatory custom library paths for DeepResonance environment
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

$PYTHON_DRIVER_DR -u src/probe_deepresonance.py --target_dir "$TARGET_DIR" --output_dir "$OUTPUT_DIR_DR"

echo "=== ALL PROBING COMPLETE === $(date)"
