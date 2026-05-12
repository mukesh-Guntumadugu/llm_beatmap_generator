#!/bin/bash
#SBATCH --job-name=qwen_generate
#SBATCH --output=logs/qwen_generate_%j.log
#SBATCH --error=logs/qwen_generate_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator
mkdir -p outputs/qwen logs

# CRITICAL: Let SLURM handle GPU isolation via --gres=gpu:1 instead of hardcoding.
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  Qwen2-Audio Director + Actor Batch Generation"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start       : $(date)"
echo "=============================================="

PYTHON=/data/mg546924/conda_envs/qwenenv/bin/python
SCRIPT=scripts/test_qwen_all_difficulties.py

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MUSIC_DIR="src/musicForBeatmap"

# Find all audio files in the music directory
mapfile -d $'\0' AUDIO_FILES < <(find "$MUSIC_DIR" -type f \( -iname "*.ogg" -o -iname "*.mp3" -o -iname "*.wav" \) -print0)

TOTAL_SONGS=${#AUDIO_FILES[@]}
echo "Found $TOTAL_SONGS songs to process."

count=1
for AUDIO_PATH in "${AUDIO_FILES[@]}"; do
    SONG_DIR=$(dirname "$AUDIO_PATH")
    SONG_BASENAME=$(basename "$AUDIO_PATH")
    SONG_NAME="${SONG_BASENAME%.*}"
    
    # Try to find an .sm or .ssc file to extract BPM
    BPM="130.0" # Default BPM
    SM_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f \( -iname "*.sm" -o -iname "*.ssc" \) | head -n 1)
    
    if [[ -n "$SM_FILE" ]]; then
        # Extract the first BPM value: e.g. #BPMS:0.000=180.000;
        EXTRACTED_BPM=$(grep -im 1 '#BPMS:' "$SM_FILE" | cut -d'=' -f2 | cut -d',' -f1 | tr -d ';\r' | awk '{print $1}')
        if [[ -n "$EXTRACTED_BPM" ]]; then
            BPM="$EXTRACTED_BPM"
        fi
    fi

    # Clean up song name for output file (remove spaces and special chars)
    CLEAN_SONG_NAME=$(echo "$SONG_NAME" | sed -e 's/[^A-Za-z0-9_-]/_/g')
    OUT_FILE="outputs/qwen/qwen_${CLEAN_SONG_NAME}_ALL_${TIMESTAMP}.ssc"

    echo ""
    echo "[$count/$TOTAL_SONGS] Generating: $SONG_NAME"
    echo "  Audio : $AUDIO_PATH"
    echo "  BPM   : $BPM"
    echo "  Output: $OUT_FILE"
    
    # Run generation
    # Using '|| true' to continue loop even if one song fails
    $PYTHON -u $SCRIPT \
        --audio "$AUDIO_PATH" \
        --bpm "$BPM" \
        --out "$OUT_FILE" || echo "WARNING: Generation failed for $SONG_NAME"

    ((count++))
done

echo ""
echo "=============================================="
echo "[OK] Qwen Batch Generation Complete!"
echo "   End : $(date)"
echo "   Output beatmaps and clusters are in 'outputs/qwen/' directory."
echo "=============================================="
