#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_onsets_%j.out
#SBATCH --error=logs/qwen_onsets_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p logs

# ── Activate Python environment ──────────────────────────────────────────────
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

MODEL_DIR="/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR="/data/mg546924/models/qwen2-audio-lora-onsets"
SERVER_PORT=8000

# ── Step 1: Start the Qwen server with LoRA weights ──────────────────────────
echo ""
echo "=== Starting Qwen2-Audio Onset Server (port $SERVER_PORT) ==="
python3 src/hpc_qwen_server.py \
    --model-dir "$MODEL_DIR" \
    --lora-dir  "$LORA_DIR" \
    --host 0.0.0.0 \
    --port $SERVER_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait up to 15 minutes for the model to load
echo "Waiting for server to become healthy..."
for i in $(seq 1 180); do
    sleep 5
    STATUS=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','false'))" 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        echo "✅ Server is ready! (waited ${i}x5s)"
        break
    fi
    echo "  Still loading... (attempt $i/180)"
done

if [ "$STATUS" != "True" ]; then
    echo "❌ Server failed to start. Check logs."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ── Step 2: Run onset extraction for ALL songs ───────────────────────────────
echo ""
echo "=== Starting Onset Extraction for ALL songs ==="
python3 scripts/extract_onsets_qwen.py \
    --server http://localhost:$SERVER_PORT

echo "=== Extraction complete at $(date) ==="

# ── Step 3: Shut down server ─────────────────────────────────────────────────
kill $SERVER_PID 2>/dev/null
echo "Job ended: $(date)"
