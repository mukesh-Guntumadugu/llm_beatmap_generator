#!/bin/bash
#SBATCH --job-name=bpm_benchmark_sweep
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/bpm_benchmark_%j.out

echo "=========================================================="
echo "Starting Unified Global BPM Full-Dataset Benchmarking"
echo "=========================================================="

DATASET_DIR="src/musicForBeatmap/Fraxtil's Arrow Arrangements"
cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

echo ""
echo "▶️ [1/5] Extracting Mathematical Ground Truth (Librosa)"
/data/mg546924/conda_envs/deepresonance_env/bin/python onsetdetection/verify_model_bpm.py --batch_dir "$DATASET_DIR" --model librosa

echo ""
echo "▶️ [2/5] Benchmarking Qwen2-Audio"
/data/mg546924/conda_envs/qwenenv/bin/python onsetdetection/verify_model_bpm.py --batch_dir "$DATASET_DIR" --model qwen

echo ""
echo "▶️ [3/5] Benchmarking MuMu-LLaMA"
/home/mg546924/.conda/envs/mumullama/bin/python onsetdetection/verify_model_bpm.py --batch_dir "$DATASET_DIR" --model mumu

echo ""
echo "▶️ [4/5] Benchmarking DeepResonance"
/data/mg546924/conda_envs/deepresonance_env/bin/python onsetdetection/verify_model_bpm.py --batch_dir "$DATASET_DIR" --model deepresonance

# echo ""
# echo "▶️ [5/5] Benchmarking Music-Flamingo (Isolated Conda Env)"
# export HF_HOME=/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints
# export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH
# /data/mg546924/music_flamingo_env/bin/python -u onsetdetection/verify_model_bpm.py --batch_dir "$DATASET_DIR" --model flamingo

echo ""
echo "✅ All 5 benchmarks completed! Check onsetdetection/ folder for the resulting CSVs."
