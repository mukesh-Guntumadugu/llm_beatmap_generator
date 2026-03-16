#!/bin/bash
#SBATCH --job-name=mumu_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=mumu_log_%j.txt

echo "=== Starting MuMu-LLaMA Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="

cd /data/mg546924/llm_beatmap_generator

# Initialize conda using the cluster's exact installation path
source /opt/shared/apps/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate mumullama

python3 extract_mumu_onsets.py

echo "=== MuMu-LLaMA Onset Detection Finished ==="
