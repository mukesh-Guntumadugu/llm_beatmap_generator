#!/bin/bash
#SBATCH --job-name=tempo_extraction
#SBATCH --partition=defq
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tempo_extraction_%j.out

echo "=========================================================="
echo " Tempo Change Extraction — Full Dataset"
echo "=========================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo ""

cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1

DATASET_DIR="src/musicForBeatmap"
OUT_CSV="onsetdetection/Tempo_Change_Analysis.csv"
WINDOW_SIZE=20    # seconds per Librosa analysis window
PYTHON=/data/mg546924/conda_envs/deepresonance_env/bin/python

echo "▶️  Running Tempo Change Extraction..."
echo "    Dataset  : $DATASET_DIR"
echo "    Output   : $OUT_CSV"
echo "    Window   : ${WINDOW_SIZE}s"
echo ""

$PYTHON onsetdetection/extract_tempo_changes.py \
    --batch_dir "$DATASET_DIR" \
    --window_size $WINDOW_SIZE \
    --out_csv "$OUT_CSV"

EXIT_CODE=$?

echo ""
echo "=========================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅  Extraction complete!  $(date)"
    echo "    Output CSV : $OUT_CSV"
    # Quick summary: count variable-tempo songs
    if [ -f "$OUT_CSV" ]; then
        TOTAL=$(tail -n +2 "$OUT_CSV" | wc -l)
        VARIABLE=$(tail -n +2 "$OUT_CSV" | awk -F',' '{print $5}' | grep -c "True")
        echo "    Songs processed   : $TOTAL"
        echo "    Variable BPM songs: $VARIABLE"
    fi
else
    echo "❌  Extraction FAILED with exit code $EXIT_CODE"
fi
echo "=========================================================="
