#!/bin/bash

# Define directory and virtual environment path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Navigate to script directory
cd "$SCRIPT_DIR"

# Check if venv exists, if not create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install dependencies if librosa is missing
if ! python3 -c "import librosa" &> /dev/null; then
    echo "Installing dependencies (librosa, numpy)..."
    pip install librosa numpy
fi

# Run the analysis script
echo "Running analysis..."
python3 analyze_beatmap.py
