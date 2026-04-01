#!/bin/bash
#SBATCH --job-name=qwen_batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=160:00:00           # Running 20 iterations back-to-back will take ~160 hours
#SBATCH --output=logs/qwen_batch_%j.out

echo "=== QWEN MEASURE-BY-MEASURE: Starting 20 Loop Iterations === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1

# Fix dynamic link paths if needed
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

PYTHON_DRIVER="/data/mg546924/conda_envs/qwenenv/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"

for ITERATION in {1..20}; do
    echo "========================================================"
    echo "=== STARTING ITERATION $ITERATION / 20 === $(date)"
    echo "========================================================"

    # Loop through all 20 song directories
    find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r SONG_DIR; do
        SONG_NAME=$(basename "$SONG_DIR")
        
        # Locate the audio file (.ogg or .mp3)
        AUDIO_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f -name "*.ogg" -o -name "*.mp3" | head -n 1)

        if [ -z "$AUDIO_FILE" ]; then
            echo "No audio file found for $SONG_NAME, skipping."
            continue
        fi

        echo "--------------------------------------------------------"
        echo "Iteration: $ITERATION | Processing Song: $SONG_NAME"
        echo "Audio: $AUDIO_FILE"
        echo "--------------------------------------------------------"

        export BENCHMARK_AUDIO="$AUDIO_FILE"
        export BENCHMARK_OUT="$SONG_DIR"
        export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator

        # Run the generator (Output automatically goes to Qwen/Wonsets_Wtempo_WBPM/ with timestamp)
        $PYTHON_DRIVER -u scripts/Qwen/qwen_measure_generator.py

    done

    echo "=== ITERATION $ITERATION DONE $(date) ==="
done

echo "=== ALL 20 ITERATIONS COMPLETE $(date) ==="
