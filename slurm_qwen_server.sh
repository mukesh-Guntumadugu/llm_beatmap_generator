#!/bin/bash
#SBATCH --job-name=qwen_beatmap
#SBATCH --partition=defq            # Default GPU partition on Ohio HPC
#SBATCH --gres=gpu:A6000:1          # 1x A6000 GPU (48GB VRAM — plenty for Qwen2-Audio-7B)
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00             # 12 hours per array task (20 songs × ~10 min = ~3.5 hrs)
#SBATCH --array=0-4                 # 5 tasks: one per difficulty (Beginner/Easy/Medium/Hard/Challenge)
#SBATCH --output=/data/mg546924/logs/qwen_beatmap_%A_%a.out   # %A=job ID, %a=array index
#SBATCH --error=/data/mg546924/logs/qwen_beatmap_%A_%a.err

# ── Map array index → difficulty ──────────────────────────────────────────────
DIFFICULTIES=("Beginner" "Easy" "Medium" "Hard" "Challenge")
DIFFICULTY=${DIFFICULTIES[$SLURM_ARRAY_TASK_ID]}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID  →  Difficulty: $DIFFICULTY"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Create logs dir if it doesn't exist
mkdir -p /data/mg546924/logs

# ── Activate Python environment ────────────────────────────────────────────────
VENV_PATH="/data/mg546924/envs/qwen_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Using venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠️  Venv not found — installing required packages on compute node..."
    export PATH="$HOME/.local/bin:$PATH"
    python3 -m pip install --user --quiet packaging psutil transformers torch torchaudio librosa fastapi uvicorn requests soundfile
    echo "✅ Packages installed."
fi

# Go to project directory
cd /data/mg546924/llm_beatmap_generator

MODEL_DIR="/data/mg546924/models/Qwen2-Audio-7B-Instruct"

# Each array task uses a different port so they don't collide
SERVER_PORT=$((8000 + SLURM_ARRAY_TASK_ID))

# ── Step 1: Start the Qwen inference server in background ─────────────────────
echo ""
echo "=== Starting Qwen2-Audio Server (port $SERVER_PORT) ==="
python3 src/hpc_qwen_server.py \
    --model-dir "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port $SERVER_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for the server to be ready — up to 10 minutes (model loading ~2-4 min on A6000)
echo "Waiting for server to become healthy..."
for i in $(seq 1 120); do
    sleep 5
    STATUS=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','false'))" 2>/dev/null)
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

# ── Step 2: Run the batch runner for this difficulty ──────────────────────────
echo ""
echo "=== Starting Batch Runner (Difficulty: $DIFFICULTY) ==="
python3 src/hpc_batch_runner.py \
    --server http://localhost:$SERVER_PORT \
    --difficulty "$DIFFICULTY"

BATCH_EXIT=$?

# ── Step 3: Cleanup ───────────────────────────────────────────────────────────
echo ""
echo "=== Batch finished (exit code: $BATCH_EXIT). Shutting down server. ==="
kill $SERVER_PID 2>/dev/null
echo "Job ended: $(date)"
