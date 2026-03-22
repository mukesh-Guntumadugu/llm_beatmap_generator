#!/bin/bash
#SBATCH --job-name=pixabay_download
#SBATCH --output=logs/pixabay_download_%j.log
#SBATCH --error=logs/pixabay_download_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=defq

# ============================================================
# Download 100 Pixabay songs into pixabay_music/ on the cluster
# Uses Pixabay API (no browser required — works on HPC)
#
# Usage:
#   sbatch slurm_download_pixabay.sh
#   sbatch slurm_download_pixabay.sh --export=ALL,PIXABAY_KEY=your_key_here
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "=============================="

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "bpenv/bin/activate" ]; then
    source bpenv/bin/activate
fi

mkdir -p logs pixabay_music

# Run downloader — API key is picked from env var PIXABAY_KEY if set
python scripts/download_pixabay_music_api.py \
    --output_dir ./pixabay_music \
    --count 100 \
    --order ec \
    ${PIXABAY_KEY:+--api_key "$PIXABAY_KEY"}

echo ""
echo "=============================="
echo "Done at: $(date)"
echo "Files  : $(ls pixabay_music/*.mp3 2>/dev/null | wc -l) MP3s in pixabay_music/"
echo "=============================="
