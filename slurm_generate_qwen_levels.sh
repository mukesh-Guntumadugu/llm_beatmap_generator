#!/bin/bash
#SBATCH --job-name=qwen_generate
#SBATCH --output=logs/qwen_generate_%j.log
#SBATCH --error=logs/qwen_generate_%j.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator
mkdir -p outputs logs

# CRITICAL: Only use 1 GPU to prevent SLURM DataParallel deadlock
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  Qwen2-Audio Director + Actor Beatmap Generation"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

PYTHON=/data/mg546924/conda_envs/qwenenv/bin/python
SCRIPT=scripts/test_qwen_all_difficulties.py

# ── Song 1: Bad Ketchup ──────────────────────────
echo ""
echo "Generating: Bad Ketchup (BPM 180)"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg" \
    --bpm 180.0 \
    --out "outputs/qwen_Bad_Ketchup_ALL.ssc"

# ── Song 2: Springtime ───────────────────────────
echo ""
echo "Generating: Springtime (BPM 145)"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3" \
    --bpm 145.0 \
    --out "outputs/qwen_Springtime_ALL.ssc"

# ── Song 3: Mecha-Tribe Assault ──────────────────
echo ""
echo "Generating: Mecha-Tribe Assault (BPM 200)"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg" \
    --bpm 200.0 \
    --out "outputs/qwen_MechaTribe_ALL.ssc"

echo ""
echo "=============================================="
echo "[OK] Qwen Generation Complete!"
echo "   End : $(date)"
echo "   Output beatmaps are in 'outputs/' directory."
echo "=============================================="
