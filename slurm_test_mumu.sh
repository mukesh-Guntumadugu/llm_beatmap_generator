#!/bin/bash
#SBATCH --job-name=test_mumu_5diff
#SBATCH --output=logs/test_mumu_5diff_%j.log
#SBATCH --error=logs/test_mumu_5diff_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "MuMu Director — All 5 Difficulty Test"
echo "Job ID : $SLURM_JOB_ID"
echo "Start  : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export HF_HOME="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts"
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))

# ── Edit these two lines to point at your test song ──
AUDIO="/data/mg546924/llm_beatmap_generator/test_audio/test_song.wav"
BPM=130.0

mkdir -p output

/data/mg546924/conda_envs/deepresonance_env/bin/python3.10 \
    scripts/test_mumu_all_difficulties.py \
    --audio "$AUDIO" \
    --bpm   "$BPM" \
    --out   "output/test_beatmap_5diff.ssc"

echo "=============================================="
echo "✅ Test Complete: $(date)"
echo "Output: output/test_beatmap_5diff.ssc"
echo "=============================================="
