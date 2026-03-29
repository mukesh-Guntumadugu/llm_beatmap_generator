#!/bin/bash
#SBATCH --job-name=DR_onset
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/mg546924/llm_beatmap_generator/logs/DR_onset_%j.out

echo "=== Starting DeepResonance Onset Inference Pipeline ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# 1. Bypass the broken Sony developer pip packages to complete PyTorch installation
cd /data/mg546924/llm_beatmap_generator/DeepResonance
grep -vE "madmom|pylast|torch|torchaudio|torchvision" valid_reqs.txt > fixed_reqs.txt
echo "Installing base PyTorch manually with correct CUDA wheels..."
/data/mg546924/conda_envs/deepresonance_env/bin/pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
echo "Completing Python environment AI installations..."
/data/mg546924/conda_envs/deepresonance_env/bin/pip install -r fixed_reqs.txt
/data/mg546924/conda_envs/deepresonance_env/bin/pip install librosa triton pytorchvideo

# 2. Download ImageBind (Huge) prerequisite if missing
cd /data/mg546924/llm_beatmap_generator/DeepResonance/ckpt/pretrained_ckpt
if [ ! -d "imagebind-huge" ]; then
    echo "Downloading ImageBind-Huge missing checkpoint (4.5GB)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nielsr/imagebind-huge
    cd imagebind-huge
    git lfs pull
fi

# Must cd into DeepResonance/code/ so config/base.yaml is found by relative path
cd /data/mg546924/llm_beatmap_generator/DeepResonance/code
/data/mg546924/conda_envs/deepresonance_env/bin/python /data/mg546924/llm_beatmap_generator/extract_deepresonance_onsets.py --ckpt_path /data/mg546924/llm_beatmap_generator/DeepResonance/ckpt

echo "Pipeline executed on $(date)."
