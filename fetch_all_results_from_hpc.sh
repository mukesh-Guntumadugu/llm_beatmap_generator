#!/bin/bash

echo "Connecting to hpc.ent.ohio.edu to fetch all Model predictions, logs, and graphs..."

# 1. Fetch Outputs Directory (CSVs, PNGs)
rsync -avz -e "ssh -o StrictHostKeyChecking=no" mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/outputs/ /Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/outputs/

# 2. Fetch Logs Directory
rsync -avz -e "ssh -o StrictHostKeyChecking=no" mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/logs/ /Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/logs/

# 3. Fetch Prediction txt/csv files inside Fraxtil's Arrangements (excluding the huge audio files)
rsync -avzm -e "ssh -o StrictHostKeyChecking=no" \
  --include="*/" \
  --include="*.csv" \
  --include="*.txt" \
  --exclude="*" \
  "mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil*Arrow*Arrangements/" \
  "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/"

echo "Done! All results and logs are now on your laptop."
