#!/bin/bash

# Array of segment durations to sweep over
CHUNK_SIZES=(20 15 10 5 2 1)

echo "================================================----"
echo "Starting Sequential Onset Extraction Sweep For Qwen"
echo "================================================----"

# Loop over the models (Qwen in this example) one chunk size at a time.
for chunk in "${CHUNK_SIZES[@]}"; do
    echo ""
    echo "▶️ Running Qwen with chunk size: ${chunk}s"
    echo "------------------------------------------------"
    
    python scripts/extract_onsets_qwen.py --chunk_sec "$chunk"
    
    echo "✅ Completed chunk size: ${chunk}s"
done

echo ""
echo "🎉 All sweeps completed successfully!"
