# Architectural Onset Probing (Flamingo & DeepResonance)

## Why are we doing this?
After mathematically probing Qwen-Audio using `probe_qwen.py`, we proved that its **Whisper** audio backbone fundamentally fails to capture rigid sparse drum rhythms accurately (it primarily tracks volume/RMS Energy). This happens because Whisper is optimized for human speech, not Music Information Retrieval (MIR).

This toolset runs the exact same strict mathematical benchmarking trial on your two other massive Multi-Modal architectures to visually see if their "brains" handle musical rhythm any better.

## The Contenders
1. **Music-Flamingo:** Uses an **AudioMAE** (Masked Autoencoder) acoustic backbone.
2. **DeepResonance:** Uses an **ImageBind** acoustic backbone linked to a Vicuna LLM. 

*(Note: MuMu-LLaMA is excluded from this benchmark temporarily until its 13GB LLaMA-2 repository weights are fully synced onto the cluster).*

## The Benchmark Files
*   `src/probe_flamingo.py`: Strips away the LLM text decoder and pulls the raw, compressed 1D latent attention directly from Flamingo's AudioMAE processor.
*   `src/probe_deepresonance.py`: Intercepts DeepResonance's generic audio pipeline.
*   `slurm_run_probe_comparison.sh`: The unified batch execution script that orchestrates the test safely on the HPC cluster so we don't accidentally blow up the 40GB `VRAM` limit by trying to load both architectures at once.

## How to Run it (Cluster)
Submit the unified Slurm batch script from the root of your project:

```bash
sbatch slurm_run_probe_comparison.sh
```

## Reviewing the Scientific Results
This job will take longer than the Qwen test because it has to safely tear down the `flamingo_env`, clear the GPU memory, and rebuild the pipeline halfway through for `deepresonance_env`.

When it is finished, download the two new directories to your local Macbook Terminal:
```bash
scp -r "mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/results_flamingo_probe_fraxtil" ./
scp -r "mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/results_dr_probe_fraxtil" ./
```

Open both `.png` charts up next to the Qwen chart. 
**The Winning Model** will be whichever architecture's Red Neural Activation line successfully spikes *only* directly on top of the green dotted tempo beats rather than erratically jumping everywhere!
