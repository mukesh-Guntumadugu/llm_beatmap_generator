#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --output=logs/qwen_onsets_%j.out
#SBATCH --error=logs/qwen_onsets_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

PYTHON="/data/mg546924/conda_envs/qwenenv/bin/python"

SERVER_PORT=8000
MODEL_DIR="/data/mg546924/models/Qwen2-Audio-7B-Instruct"
LORA_DIR="/data/mg546924/models/qwen2-audio-lora-onsets"

echo "=== Starting Qwen2-Audio Server for ONSETS (port $SERVER_PORT) ==="
$PYTHON src/hpc_qwen_server.py \
    --model-dir "$MODEL_DIR" \
    --lora-dir "$LORA_DIR" \
    --host 0.0.0.0 \
    --port $SERVER_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to become healthy..."
for i in $(seq 1 60); do
    sleep 5
    STATUS=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','false'))" 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        echo "✅ Server is ready!"
        break
    fi
done

if [ "$STATUS" != "True" ]; then
    echo "❌ Server failed to start. Check logs."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "=== Extracting Exact Millisecond Onsets ==="
# Run the extraction python script for all 20 songs, 5-second chunk size
$PYTHON scripts/extract_onsets_qwen.py --server http://localhost:$SERVER_PORT --chunk_sec 5.0

echo "=== Shutting down server. ==="
kill $SERVER_PID 2>/dev/null
echo "✅ Job Complete!"
