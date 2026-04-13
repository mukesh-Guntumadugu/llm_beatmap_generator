#!/bin/bash
#SBATCH --job-name=train_hierarchical_director
#SBATCH --output=logs/train_hierarchical_%j.log
#SBATCH --error=logs/train_hierarchical_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

set -e
mkdir -p /data/mg546924/llm_beatmap_generator/logs
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "  Hierarchical Director Training (Phase 3)"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "  Strategy    : Cluster-Token SFT (Director/Actor)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# Verify the required inputs exist before starting
echo ""
echo "── Pre-flight checks ──"

if [ ! -f "hierarchical_sft_dataset/hierarchical_train.jsonl" ]; then
    echo "❌ ERROR: hierarchical_train.jsonl not found!"
    echo "   Run: python scripts/prepare_hierarchical_sft.py"
    exit 1
fi
echo "  ✅ hierarchical_train.jsonl found"

if [ ! -f "scripts/cluster_to_patterns_tokens.txt" ]; then
    echo "❌ ERROR: cluster_to_patterns_tokens.txt not found!"
    echo "   Run: python scripts/build_sub_decoder_dict.py"
    exit 1
fi
echo "  ✅ cluster_to_patterns_tokens.txt found"

WINDOW_COUNT=$(wc -l < hierarchical_sft_dataset/hierarchical_train.jsonl)
TOKEN_COUNT=$(wc -l < scripts/cluster_to_patterns_tokens.txt)
echo "  ✅ Training windows : $WINDOW_COUNT"
echo "  ✅ Cluster tokens   : $TOKEN_COUNT"
echo ""

/data/mg546924/conda_envs/qwenenv/bin/python scripts/train_hierarchical_mumu.py

echo ""
echo "=============================================="
echo "✅ Hierarchical Director Training Complete!"
echo "   End : $(date)"
echo "=============================================="
