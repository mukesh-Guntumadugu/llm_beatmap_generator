#!/bin/bash

# Array of segment durations to sweep over
CHUNK_SIZES=(20 15 10 5 2 1)

echo "================================================----"
echo "Starting Sequential Onset Extraction Sweep For Flamingo"
echo "================================================----"

for chunk in "${CHUNK_SIZES[@]}"; do
    echo ""
    echo "▶️ Running Flamingo with chunk size: ${chunk}s"
    echo "------------------------------------------------"
    
    python onsetdetection/extract_onsets_flamingo.py --chunk_sec "$chunk"
    
    echo "✅ Completed Flamingo chunk size: ${chunk}s"
done

echo ""
echo "🎉 All Flamingo sweeps completed successfully!"
