#!/bin/bash
#SBATCH --job-name=qwen_beatmap_gen
#SBATCH --output=logs/qwen_gen_%j.log
#SBATCH --error=logs/qwen_gen_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Initialize environment
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# Ensure output directory exists
mkdir -p output/qwen_beatmaps

echo "======================================"
echo "Starting Qwen Hierarchical Generation"
echo "======================================"

# Run the test script on a sample audio file from your dataset
# Adjust the path to an audio file you want to generate a beatmap for
AUDIO_FILE="/data/mg546924/llm_beatmap_generator/sft_dataset_pixabay/audio/487408_None_003.wav"
OUTPUT_FILE="output/qwen_beatmaps/test_487408.ssc"
BPM=130.0 # Default fallback, adjust if needed

# Run generation for all difficulties using the specific qwen python environment
/data/mg546924/conda_envs/qwenenv/bin/python scripts/test_qwen_all_difficulties.py \
    --audio "$AUDIO_FILE" \
    --bpm "$BPM" \
    --out "$OUTPUT_FILE"

echo "======================================"
echo "Generation complete! Saved to $OUTPUT_FILE"
echo "======================================"
