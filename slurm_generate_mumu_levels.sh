#!/bin/bash
#SBATCH --job-name=generate_mumu_levels
#SBATCH --output=logs/generate_mumu_levels_%j.log
#SBATCH --error=logs/generate_mumu_levels_%j.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

set -e
mkdir -p /data/mg546924/llm_beatmap_generator/logs
mkdir -p /data/mg546924/llm_beatmap_generator/outputs
cd /data/mg546924/llm_beatmap_generator

export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  MuMu-LLaMA Director & Actor Level Generation"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

# Song 1: Bad Ketchup
echo ""
echo "Generating Level for: Bad Ketchup"
/data/mg546924/conda_envs/qwenenv/bin/python scripts/generate_hierarchical_beatmap.py \
    --audio "src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg" \
    --bpm 180.0 \
    --out "outputs/mumu_generated_Bad_Ketchup.ssc"

# Song 2: Springtime
echo ""
echo "Generating Level for: Springtime"
/data/mg546924/conda_envs/qwenenv/bin/python scripts/generate_hierarchical_beatmap.py \
    --audio "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3" \
    --bpm 180.0 \
    --out "outputs/mumu_generated_Springtime.ssc"

# Song 3: Mecha-Tribe Assault
echo ""
echo "Generating Level for: Mecha-Tribe Assault"
/data/mg546924/conda_envs/qwenenv/bin/python scripts/generate_hierarchical_beatmap.py \
    --audio "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg" \
    --bpm 180.0 \
    --out "outputs/mumu_generated_MechaTribe.ssc"

echo ""
echo "=============================================="
echo "✅ Level Generation Complete!"
echo "   End : $(date)"
echo "   Output beatmaps are in 'outputs/' directory."
echo "=============================================="
