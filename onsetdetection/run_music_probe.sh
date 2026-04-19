#!/bin/bash
#SBATCH --job-name=music_probe
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=node002
#SBATCH --output=music_probe_%j.txt

echo "=== Music Knowledge Probe — All Models ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Time   : $(date)"
echo "=========================================="

cd /data/mg546924/llm_beatmap_generator
source /home/mg546924/.conda/etc/profile.d/conda.sh

# ── MuMu-LLaMA (needs mumullama env) ────────────────────────────
echo ""
echo ">>> Probing MuMu-LLaMA..."
conda run -n mumullama python3 -u onsetdetection/probe_model_music_knowledge.py --model mumu

# ── Gemini (API — no GPU, same env is fine) ───────────────────────
echo ""
echo ">>> Probing Gemini..."
conda run -n mumullama python3 -u onsetdetection/probe_model_music_knowledge.py --model gemini

# ── Qwen2-Audio (needs its own env if separate, else mumullama) ───
echo ""
echo ">>> Probing Qwen2-Audio..."
conda run -n mumullama python3 -u onsetdetection/probe_model_music_knowledge.py --model qwen

# ── DeepResonance ─────────────────────────────────────────────────
echo ""
echo ">>> Probing DeepResonance..."
CONDA_ENV="deepresonance"
if conda env list | grep -q "$CONDA_ENV"; then
    conda run -n $CONDA_ENV python3 -u onsetdetection/probe_model_music_knowledge.py --model deepresonance
else
    conda run -n mumullama python3 -u onsetdetection/probe_model_music_knowledge.py --model deepresonance
fi

echo ""
echo "=== Probe Finished: $(date) ==="
