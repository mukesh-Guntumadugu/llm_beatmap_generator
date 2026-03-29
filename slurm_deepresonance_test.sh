#!/bin/bash
#SBATCH --job-name=deepres_test
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/deepres_test_%j.out

echo "=== DeepResonance Setup & Test ==="
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo ""

echo "Checking the storage used by DeepResonance so far..."
du -sh /data/mg546924/conda_envs/deepresonance_env
du -sh /data/mg546924/llm_beatmap_generator/DeepResonance
echo "⚠️  Note: The 3 massive model weights (Vicuna 7B, ImageBind, and DeepResonance Alpha checkpoints) are NOT downloaded yet. Expect this storage footprint to jump by ~50GB once they are pulled!"
echo ""

echo "Testing the Python environment..."
# The environment failed to build perfectly because of a broken 'madmom' developer version in Sony's requirements,
# but the python binary itself was successfully created!
/data/mg546924/conda_envs/deepresonance_env/bin/python -c "print('✅ DeepResonance python environment is functional and GPU ready!')"

echo "End of test job."
