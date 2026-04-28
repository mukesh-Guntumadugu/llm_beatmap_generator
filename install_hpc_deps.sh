#!/bin/bash
echo "=============================================="
echo "Installing AudioFlamingo Dependencies"
echo "=============================================="

# AudioFlamingo requires the bleeding-edge transformers from GitHub.
# Since compute nodes lack internet, we must install it here on the login node.

echo "Upgrading transformers in deepresonance_env..."
/data/mg546924/conda_envs/deepresonance_env/bin/python3.10 -m pip install --upgrade pip
/data/mg546924/conda_envs/deepresonance_env/bin/python3.10 -m pip install --upgrade git+https://github.com/huggingface/transformers accelerate

echo "✅ Installation complete!"
