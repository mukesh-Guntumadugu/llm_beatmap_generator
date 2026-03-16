#!/bin/bash
#SBATCH --job-name=qwen_beatmap
#SBATCH --partition=defq            # Default GPU partition on Ohio HPC
#SBATCH --gres=gpu:A6000:1          # 1x A6000 GPU (48GB VRAM — plenty for Qwen2-Audio-7B)
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00             # 12 hours per array task (20 songs × ~10 min = ~3.5 hrs)
#SBATCH --array=1-1                 # TEST MODE: only Easy (change to 0-4 for all difficulties)
#SBATCH --output=logs/qwen_beatmap_%A_%a.out
#SBATCH --error=logs/qwen_beatmap_%A_%a.err

# ── Map array index → difficulty ──────────────────────────────────────────────
DIFFICULTIES=("Beginner" "Easy" "Medium" "Hard" "Challenge")
DIFFICULTY=${DIFFICULTIES[$SLURM_ARRAY_TASK_ID]}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID  →  Difficulty: $DIFFICULTY"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Create logs dir (must exist before SLURM writes output — runs after job start so logs/ must be pre-created)
mkdir -p logs

# ── Activate Python environment ────────────────────────────────────────────────
# Use the conda env that already has all packages installed (same as run_qwen.sh)
CONDA_PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"
CONDA_BIN="/data/mg546924/conda_envs/qwenenv/bin"
CONDA_PIP="/data/mg546924/conda_envs/qwenenv/bin/pip"

if [ -f "$CONDA_PYTHON" ]; then
    echo "Using conda env: $CONDA_BIN"
    export PATH="$CONDA_BIN:$PATH"
    export PYTHONNOUSERSITE=1
else
    echo "❌ Conda env not found at $CONDA_BIN — aborting."
    exit 1
fi

# ── Install any missing packages directly into the conda env ─────────────────
echo "Ensuring required packages are installed in conda env..."
$CONDA_PIP install --quiet uvicorn fastapi pydantic soundfile librosa requests lm-format-enforcer
echo "✅ Package check done."

# Go to project directory
cd /data/mg546924/llm_beatmap_generator

# Must export PYTHONPATH so python can find the `src` module imports
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

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
# TEST MODE: set TEST_SONG to a song name to test on ONE song only.
# Set to "" to run on ALL songs.
TEST_SONG="Bad Ketchup"

echo ""
echo "=== Starting Batch Runner (Difficulty: $DIFFICULTY) ==="

if [ -n "$TEST_SONG" ]; then
    echo "TEST MODE: Running on single song: $TEST_SONG"
    python3 src/hpc_batch_runner.py \
        --server http://localhost:$SERVER_PORT \
        --difficulty "$DIFFICULTY" \
        --job-id "$SLURM_JOB_ID" \
        --song "$TEST_SONG"
else
    python3 src/hpc_batch_runner.py \
        --server http://localhost:$SERVER_PORT \
        --difficulty "$DIFFICULTY" \
        --job-id "$SLURM_JOB_ID"
fi

BATCH_EXIT=$?

# ── Step 3: Cleanup ───────────────────────────────────────────────────────────
echo ""
echo "=== Batch finished (exit code: $BATCH_EXIT). Shutting down server. ==="
kill $SERVER_PID 2>/dev/null
echo "Job ended: $(date)"
