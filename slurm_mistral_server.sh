#!/bin/bash
#SBATCH --job-name=mistral_beatmap
#SBATCH --partition=defq            # Default GPU partition on Ohio HPC
#SBATCH --gres=gpu:A6000:1          # 1x A6000 GPU (48GB VRAM — fits Mistral-7B fp16 ~14GB)
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00             # 12 hours max
#SBATCH --output=logs/mistral_beatmap_%j.out
#SBATCH --error=logs/mistral_beatmap_%j.err

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p logs

# Activate Python venv — falls back to --user install if venv not found
VENV_PATH="/data/mg546924/envs/mistral_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Using venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠️  Venv not found — installing packages on compute node..."
    export PATH="$HOME/.local/bin:$PATH"
    python3 -m pip install --user --quiet \
        transformers torch torchaudio librosa \
        fastapi uvicorn requests soundfile numpy
    echo "✅ Packages installed."
fi

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

MODEL_DIR="/data/mg546924/models/Mistral-7B-Instruct-v0.3"
SERVER_PORT=8001

# ── (Optional) Download model if not present ──────────────────────────────────
# Uncomment and run once on a login node that has internet access:
# python3 -c "
# from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', cache_dir='$MODEL_DIR')
# AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', cache_dir='$MODEL_DIR')
# print('Model downloaded.')
# "

# ── Step 1: Start the Mistral inference server in background ──────────────────
echo ""
echo "=== Starting Mistral-7B Server (port $SERVER_PORT) ==="
python3 src/hpc_mistral_server.py \
    --model-dir "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port $SERVER_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait up to 10 minutes for the server to become healthy
echo "Waiting for server to load model (~2-5 min for 7B)..."
for i in $(seq 1 120); do
    sleep 5
    STATUS=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','false'))" 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        echo "✅ Server is ready! (waited ${i}x5s)"
        break
    fi
    echo "  Still loading... (attempt $i/120)"
done

if [ "$STATUS" != "True" ]; then
    echo "❌ Server failed to start within 10 minutes. Check logs."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ── Step 2: Run the Mistral batch runner ──────────────────────────────────────
echo ""
echo "=== Starting Mistral Batch Runner ==="
python3 src/hpc_mistral_batch_runner.py --server http://localhost:$SERVER_PORT

BATCH_EXIT=$?

# ── Step 3: Cleanup ───────────────────────────────────────────────────────────
echo ""
echo "=== Batch finished (exit code: $BATCH_EXIT). Shutting down server. ==="
kill $SERVER_PID 2>/dev/null
echo "Job ended: $(date)"
