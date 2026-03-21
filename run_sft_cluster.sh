#!/bin/bash
#SBATCH --job-name=qwen_sft
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen_sft_%j.out
#SBATCH --error=logs/qwen_sft_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p logs

# ── Activate Python environment ─────────────────────────────────────────────
CONDA_BIN="/data/mg546924/conda_envs/qwenenv/bin"
if [ -f "$CONDA_BIN/python" ]; then
    echo "Using conda env: $CONDA_BIN"
    export PATH="$CONDA_BIN:$PATH"
    export PYTHONNOUSERSITE=1
else
    echo "❌ Conda env not found at $CONDA_BIN — aborting."
    exit 1
fi

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# Run SFT Trainer Script
echo "Starting Qwen2-Audio SFT with QLoRA..."
python3 scripts/train_qwen2_audio_lora.py

echo "Job ended: $(date)"
