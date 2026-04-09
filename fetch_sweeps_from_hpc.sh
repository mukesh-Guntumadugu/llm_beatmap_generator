#!/bin/bash

# This script pulls all predicted Qwen sweep files (.csv) from the HPC cluster 
# down to your laptop maintaining the exact same directory structure.
# You will be prompted to enter your HPC password.

echo "Connecting to hpc.ent.ohio.edu to fetch missing Qwen sweeps (.csv files)..."

rsync -avzm -e "ssh -o StrictHostKeyChecking=no" \
  --include="*/" \
  --include="*.csv" \
  --exclude="*" \
  "mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/" \
  "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/"

echo "Done! The data should now be locally available."
