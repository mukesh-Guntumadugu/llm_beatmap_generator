#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=qwen_log_%j.txt

echo "=== Starting Qwen2-Audio Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================="

cd /data/mg546924/llm_beatmap_generator

# Initialize conda using the cluster's exact installation path
source /opt/shared/apps/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate qwenenv

python3 extract_qwen_onsets.py

echo "=== Qwen Onset Detection Finished ==="
