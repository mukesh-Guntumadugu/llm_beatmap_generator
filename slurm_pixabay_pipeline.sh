#!/bin/bash
#SBATCH --job-name=pixabay_qwen_pipeline
#SBATCH --output=logs/pixabay_pipeline_%j.log
#SBATCH --error=logs/pixabay_pipeline_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=defq
#SBATCH --gres=gpu:1          # needed for Qwen training step

# ============================================================
# Full pipeline: Download 100 Pixabay songs → Librosa onsets
#                → SFT dataset → Qwen2-Audio fine-tuning
#
# Prerequisites (run LOCALLY first):
#   python scripts/collect_pixabay_urls.py --count 100 --output pixabay_urls.json
#   # Then copy pixabay_urls.json to the cluster alongside your code
#
# Submit:
#   sbatch slurm_pixabay_pipeline.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start       : $(date)"
echo "Working dir : $(pwd)"
echo "=============================================="

# Activate virtualenv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "bpenv/bin/activate" ]; then
    source bpenv/bin/activate
fi

mkdir -p logs pixabay_music sft_dataset_pixabay

# ── STEP A: Download MP3s from collected URLs ───────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP A: Downloading songs from pixabay_urls.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "pixabay_urls.json" ]; then
    echo "❌ pixabay_urls.json not found!"
    echo "   Run locally: python scripts/collect_pixabay_urls.py --count 100"
    echo "   Then copy pixabay_urls.json to the cluster."
    exit 1
fi

python scripts/download_from_urls.py \
    --urls_file pixabay_urls.json \
    --output_dir ./pixabay_music \
    --delay 0.2

echo "Songs downloaded: $(ls pixabay_music/*.mp3 2>/dev/null | wc -l)"

# ── STEP B: Librosa onset extraction ───────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP B: Extracting Librosa onsets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/extract_librosa_onsets_generic.py \
    --audio_dir ./pixabay_music

echo "Onset CSVs: $(ls pixabay_music/original_onsets_*.csv 2>/dev/null | wc -l)"

# ── STEP C: Build SFT dataset ───────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP C: Building SFT dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/prepare_sft_dataset_generic.py \
    --audio_dir ./pixabay_music \
    --output_dir ./sft_dataset_pixabay

echo "Training samples: $(wc -l < sft_dataset_pixabay/dataset.jsonl 2>/dev/null || echo 0)"

# ── STEP D: Fine-tune Qwen2-Audio ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP D: Fine-tuning Qwen2-Audio with LoRA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Override dataset path to use the new Pixabay dataset
export DATASET_OVERRIDE="$(pwd)/sft_dataset_pixabay/dataset.jsonl"

# The training script already has cluster paths set — just run it
python scripts/train_qwen2_audio_lora.py

echo ""
echo "=============================================="
echo "✅ Pipeline complete!"
echo "   Songs       : pixabay_music/ ($(ls pixabay_music/*.mp3 2>/dev/null | wc -l) files)"
echo "   Onset CSVs  : $(ls pixabay_music/original_onsets_*.csv 2>/dev/null | wc -l) files"
echo "   Dataset     : sft_dataset_pixabay/dataset.jsonl"
echo "   Model       : /data/mg546924/models/qwen2-audio-lora-onsets"
echo "   Finished    : $(date)"
echo "=============================================="
