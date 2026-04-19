#!/bin/bash
#SBATCH --job-name=music_probe
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=node002
#SBATCH --output=music_probe_%j.txt

echo "=== Music Knowledge Probe — All Models ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Time   : $(date)"
echo "=========================================="

cd /data/mg546924/llm_beatmap_generator

# Redirect user site-packages to /data (home quota is full)
export PYTHONUSERBASE=/data/mg546924/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

MUMU_PY=/home/mg546924/.conda/envs/mumullama/bin/python
DEEPRES_PY=/data/mg546924/conda_envs/deepresonance_env/bin/python

echo ""
echo ">>> Probing MuMu-LLaMA..."
$MUMU_PY -u onsetdetection/probe_model_music_knowledge.py --model mumu

echo ""
echo ">>> Probing Gemini (API — no GPU needed)..."
$MUMU_PY -u onsetdetection/probe_model_music_knowledge.py --model gemini

echo ""
echo ">>> Probing Qwen2-Audio..."
$MUMU_PY -u onsetdetection/probe_model_music_knowledge.py --model qwen

echo ""
echo ">>> Probing DeepResonance..."
$DEEPRES_PY -u onsetdetection/probe_model_music_knowledge.py --model deepresonance

echo ""
echo ">>> Probing Music-Flamingo..."
$DEEPRES_PY -u onsetdetection/probe_model_music_knowledge.py --model flamingo

echo ""
echo "=== Probe Finished: $(date) ==="
