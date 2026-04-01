#!/bin/bash
#SBATCH --job-name=flam_batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=160:00:00
#SBATCH --output=logs/flamingo_batch_%j.out

echo "=== FLAMINGO MEASURE-BY-MEASURE: Starting 20 Loop Iterations === $(date)"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1

PYTHON_DRIVER="/data/mg546924/conda_envs/flamingo_env/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"

for ITERATION in {1..20}; do
    echo "========================================================"
    echo "=== STARTING ITERATION $ITERATION / 20 === $(date)"
    echo "========================================================"

    find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r SONG_DIR; do
        SONG_NAME=$(basename "$SONG_DIR")
        AUDIO_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f -name "*.ogg" -o -name "*.mp3" | head -n 1)

        if [ -z "$AUDIO_FILE" ]; then continue; fi

        echo "Iteration: $ITERATION | Processing Song: $SONG_NAME"
        export BENCHMARK_AUDIO="$AUDIO_FILE"
        export BENCHMARK_OUT="$SONG_DIR"
        export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator

        $PYTHON_DRIVER -u scripts/Flamingo/flamingo_measure_generator.py
    done
done
echo "=== ALL 20 ITERATIONS COMPLETE $(date) ==="
