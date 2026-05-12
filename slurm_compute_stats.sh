#!/bin/bash
#SBATCH --job-name=compute_dataset_stats
#SBATCH --output=logs/compute_stats_%j.log
#SBATCH --error=logs/compute_stats_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
# No GPU needed for parsing text files, so we don't request --gres=gpu

set -e
cd /data/mg546924/llm_beatmap_generator
mkdir -p outputs logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

echo "=============================================="
echo "  Extracting Dataset Statistics for Paper"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

PYTHON=/data/mg546924/conda_envs/qwenenv/bin/python
SCRIPT=scripts/compute_dataset_stats.py

$PYTHON -u $SCRIPT

echo ""
echo "=============================================="
echo "[OK] Statistics Extraction Complete!"
echo "   End : $(date)"
echo "   Look for the markdown tables in the outputs/ directory."
echo "=============================================="
