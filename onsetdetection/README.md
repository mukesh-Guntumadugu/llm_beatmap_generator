# Multimodal Onset Detection

This directory houses the standalone onset extraction scripts for the open-source evaluation models (Qwen, MuMu-LLaMA, DeepResonance, Flamingo), along with baseline reference scripts (Gemini, Librosa).

## The Anti-Hallucination Chunking Strategy
When generating exact timestamp floating-point numbers on raw audio sequences (via zero-shot cross-attention), autoregressive LLMs commonly fall victim to "Continuous Counting-Loop Hallucinations" (e.g. perpetually spitting out `[150.2, 150.3, 150.4]`). This happens exclusively on long ~3-minute songs because over immense token horizons, textual auto-completion overwhelms their audio grounding.

To mathematically eliminate this while simultaneously preserving their **raw zero-shot multimodal listening capabilities**:
- **15-Second Chunking**: Every file is sliced on the fly via `soundfile/librosa` into 15.0 second discrete boundary `.wav` clips. 
- **Context Constriction**: Model attention mechanism only deals with chunks tight enough to retain total spatial memory, avoiding the need for hybrid mathematical retrieval setups.
- **Drift Gates**: Mathematical checks array identical sequential deltas; if a loop hallucination does accidentally trigger, it is purged automatically and the single 15s chunk resets + retries up to 3 times invisibly.
- **Aggregation**: Outputs are instantly mapped from their relative internal timestamps (`10.4s`) across the base (`+15s`) to perfectly reconstruct a continuous multi-minute CSV dataset string!

## Executing Batch Sweep Files
All matching SLURM `.sh` scripts and batch runners used during inference sweeps have been moved securely alongside the `.py` files inside this directory. Their internal dependencies correctly link relatively back to the main generator codebase!
