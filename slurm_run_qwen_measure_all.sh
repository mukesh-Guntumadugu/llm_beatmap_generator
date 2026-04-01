#!/bin/bash
#SBATCH --job-name=qwen_batch
#SBATCH --array=1-20              # This spawns 20 independent jobs (iterations)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00           # Each job takes ~8 hours for all 20 songs
#SBATCH --output=logs/qwen_batch_%A_iter_%a.out

ITERATION=$SLURM_ARRAY_TASK_ID

echo "=== QWEN MEASURE-BY-MEASURE: Iteration $ITERATION / 20 === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1

# Fix dynamic link paths if needed
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

PYTHON_DRIVER="/data/mg546924/conda_envs/qwenenv/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"

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
