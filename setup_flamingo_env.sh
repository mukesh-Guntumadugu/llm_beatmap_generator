#!/bin/bash
# ============================================================
# setup_flamingo_env.sh
# Creates a dedicated conda env for Music-Flamingo training.
# Run ONCE on HPC login node:
#   bash setup_flamingo_env.sh
# ============================================================

set -e

ENV_NAME="flamingo_env"
CONDA_BASE="/data/mg546924/conda_envs"
ENV_PATH="${CONDA_BASE}/${ENV_NAME}"
PYTHON_BIN="${ENV_PATH}/bin/python"
PIP="${ENV_PATH}/bin/pip"

echo "=============================================="
echo "Creating isolated flamingo_env"
echo "Path: ${ENV_PATH}"
echo "=============================================="

# ── Create fresh env ──
if [ -d "${ENV_PATH}" ]; then
    echo "⚠️  ${ENV_PATH} already exists. Skipping creation."
else
    conda create -y -p "${ENV_PATH}" python=3.10
    echo "✅ Conda env created."
fi

# ── Core packages ──
echo "Installing PyTorch..."
${PIP} install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing latest transformers (AudioFlamingo3 requires 4.40+)..."
${PIP} install "transformers>=4.40.0"

echo "Installing remaining dependencies..."
${PIP} install \
    peft==0.10.0 \
    accelerate \
    librosa \
    soundfile \
    tqdm \
    huggingface_hub \
    datasets \
    bitsandbytes \
    scipy \
    numpy

echo ""
echo "=============================================="
echo "Verifying AudioFlamingo3 import..."
${PYTHON_BIN} -c "
from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Processor
print('✅ AudioFlamingo3 import OK')
import transformers
print(f'   transformers version: {transformers.__version__}')
"
echo "=============================================="
echo "✅ flamingo_env ready. Submit with: sbatch slurm_train_flamingo.sh"
