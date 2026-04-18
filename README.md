# 🎵 LLM Beatmap Generator

A research project that uses Large Language Models (LLMs) and multimodal AI to automatically generate **StepMania beatmap onset timestamps** from audio files. The project evaluates how well different AI models can detect musical events (onsets/transients) from raw audio and produce rhythm game charts.

---

## 🎯 Project Goal

Given a collection of songs from Fraxtil's Arrow Arrangements (20 tracks), automatically detect musical onset events in milliseconds using AI models and generate StepMania-compatible beatmap data — replacing manual human charting with automated LLM inference.

---

## 🤖 Models Currently In Use

### ✅ DeepResonance *(primary model — running on Ohio HPC)*
- **What it is:** A multimodal music LLM built on LLaMA (Vicuna 7B) + ImageBind audio encoder
- **Weights:** Available — stored at `/data/mg546924/llm_beatmap_generator/DeepResonance/ckpt/`
- **How it runs:** Ohio HPC cluster via Slurm on NVIDIA A6000 GPUs
- **Scripts:**
  - `extract_deepresonance_onsets.py` — batch onset extraction (1/5/10/15/20 second audio chunks)
  - `slurm_run_deepresonance.sh` — Slurm batch submission script
  - `chat_deepresonance.py` — interactive chat interface
  - `CHAT_DEEPRESONANCE_README.md` — full usage guide for the chat interface
- **Output:** `DeepResonance_onsets_[SongName]_[timestamp]_[chunk]sec.csv`

### ✅ Qwen LoRA *(secondary model — also running on Ohio HPC)*
- **What it is:** Qwen audio model fine-tuned with LoRA adapters for music understanding
- **Weights:** Available on cluster
- **Scripts:** `extract_qwen_onsets.py`, `run_qwen.sh`, `slurm_qwen_sequential.sh`
- **Output:** `Qwen_LoRA_onsets_[SongName]_[timestamp].txt` files in each song's `qwen_onsets/` folder

### ✅ Google Gemini Pro *(cloud API)*
- **What it is:** Google's multimodal API used for audio description and onset analysis
- **Requirement:** `GOOGLE_API_KEY` environment variable
- **Script:** `extract_gemini_onsets.py`
- **Output:** `gemini_pro_alignment_results.csv`

### ✅ Librosa *(traditional signal processing baseline)*
- **What it is:** Python audio analysis library — rule-based onset detection, no AI
- **Script:** `extract_librosa_onsets.py`
- **Purpose:** Provides a ground-truth baseline to compare LLM models against

### ✅ Music-Flamingo *(tested on cluster)*
- **What it is:** A multimodal music-language model fine-tuned for music understanding and description
- **Weights:** Available at `Music-Flamingo/`
- **Script:** `src/music_flamingo_interface.py`
- **Status:** Working on the cluster

### ✅ MuMu-LLaMA *(tested on cluster)*
- **What it is:** Music understanding multimodal LLaMA model from the MuMu dataset
- **Weights:** You must manually download the MuMu-LLaMA repository and weights from the internet and place them in the root directory under a folder called `MuMu-LLaMA/`. The codebase expects the inner path to be `MuMu-LLaMA/MuMu-LLaMA/`.
- **Scripts:** `extract_mumu_onsets.py`, `run_mumu.sh`
- **Logs:** `mumu_log_*.txt`
  
 ### ✅ DeepSeek (DeepSeek-V3 )
- **Why not:** DeepSeek has an audio as input, so we are giving it a shot.
- **Status:** trying WIP for music/audio tasks
- **Note:** It looks like some blogs are saying interesting things, we are WIP

---

## ❌ Models Tried But Could Not Use

### ✗ Spotify Basic Pitch / Spotify Audio Research Models
- **Why not:** Spotify does not publicly release model weights for their core audio intelligence models
- **Status:** Weights not available — cannot run locally or on cluster
- **Alternative used:** Librosa for traditional onset detection


### ✗ SALMONN
- **Why not:** Model directory exists (`SALMONN/`) but weights were too large to fully download and configure on the Ohio cluster storage quota
- **Status:** Attempted — setup incomplete

### ✗ LLark
- **Why not:** LLark requires proprietary audio features from the original paper's training pipeline that are not publicly available
- **Status:** Directory exists but inference is not reproducible without original training artifacts

---

## 🏗️ Project Structure

```
llm_beatmap_generator/
├── extract_deepresonance_onsets.py   # DeepResonance batch onset detector (chunks)
├── extract_qwen_onsets.py            # Qwen LoRA onset detector
├── extract_gemini_onsets.py          # Gemini API onset detector
├── extract_librosa_onsets.py         # Librosa baseline onset detector
├── extract_mumu_onsets.py            # MuMu-LLaMA onset detector
├── chat_deepresonance.py             # Interactive DeepResonance chat terminal
├── CHAT_DEEPRESONANCE_README.md      # How to chat with DeepResonance
├── score_onset_detection.py          # Evaluate onset quality vs ground truth
├── slurm_run_deepresonance.sh        # Slurm script for DeepResonance batch jobs
├── slurm_qwen_sequential.sh          # Slurm script for Qwen batch jobs
├── DeepResonance/                    # DeepResonance model code + weights
├── MuMu-LLaMA/                       # MuMu-LLaMA model (weights available)
├── SALMONN/                          # SALMONN (weights incomplete)
├── Music-Flamingo/                   # Music-Flamingo (env conflicts)
├── src/
│   └── musicForBeatmap/
│       └── Fraxtil's Arrow Arrangements/
│           ├── Bad Ketchup/
│           │   ├── Bad Ketchup.ogg                    # Original audio
│           │   ├── DeepResonance_onsets_*_5sec.csv    # AI onset predictions
│           │   └── qwen_onsets/                       # Qwen predictions
│           └── [19 more songs...]
└── logs/                             # Slurm job logs (DR_onset_*.out)
```

---

## 🧪 Benchmark: Test All Models on One Song

To quickly compare how all open-source models perform on **Bad Ketchup** (one song), use the unified benchmark:

```bash
# Submit the all-models benchmark job to the cluster
sbatch --nodelist=node009 slurm_test_all_models.sh

# Watch live output
tail -f logs/all_models_test_<JOBID>.out
```

This runs all 5 open-source models in sequence on `Bad Ketchup.ogg` and prints a summary table:

```
BENCHMARK SUMMARY — Bad Ketchup onset detection
================================================
  ✅  Librosa (it's a Python package)              →  312 onsets
  ✅  DeepResonance        →  113 onsets
  ✅  Qwen                 →  85 onsets
  ✅  MuMu-LLaMA          →  47 onsets
  ✅  Music-Flamingo       →  92 onsets
```

---

## 📋 Batch Scripts Reference

| Script | Purpose | Command |
|--------|---------|---------|
| `slurm_test_all_models.sh` | Run ALL open-source models on Bad Ketchup (benchmark) | `sbatch slurm_test_all_models.sh` |
| `slurm_run_deepresonance.sh` | Run DeepResonance on all 20 songs (full batch) | `sbatch slurm_run_deepresonance.sh` |
| `run_flamingo.sh` | Run Music-Flamingo on all 23 audio files | `sbatch run_flamingo.sh` |
| `run_mumu.sh` | Run MuMu-LLaMA on all songs | `sbatch run_mumu.sh` |
| `slurm_qwen_sequential.sh` | Run Qwen on all songs | `sbatch slurm_qwen_sequential.sh` |
| `slurm_run_qwen_measure.sh` | Run Qwen line-by-line (beat-by-beat) with Onsets, BPM, and Tempo features included | `sbatch slurm_run_qwen_measure.sh` |
| `run_onset_extraction.sh` | Run Librosa baseline on all songs | `bash run_onset_extraction.sh` |

### 🎛️ Onset Size Sweeping
We have implementations to sweep songs parsing discrete subsets (`20s, 15s, 10s, 5s, 2s, 1s`) sequentially handling latency analytics correctly.
| Script | Purpose | Command |
|--------|---------|---------|
| `slurm_sweep_onsets.sh` | Generate subset sweeps natively for Qwen server logic | `sbatch slurm_sweep_onsets.sh` |
| `slurm_sweep_flamingo.sh` | Generate subset sweeps iterating Flamingo GPU | `sbatch slurm_sweep_flamingo.sh` |

> All Slurm jobs can be monitored with `squeue -u $USER` and `tail -f logs/<logfile>.out`

---



### DeepResonance Batch Job
```bash
ssh mg546924@hpc.ent.ohio.edu
cd /data/mg546924/llm_beatmap_generator
sbatch --nodelist=node009 slurm_run_deepresonance.sh
squeue -u $USER   # monitor job
```

### DeepResonance Interactive Chat
```bash
srun -p defq --gres=gpu:1 --pty bash
/data/mg546924/conda_envs/deepresonance_env/bin/python chat_deepresonance.py
```
> See `CHAT_DEEPRESONANCE_README.md` for full instructions.

---

### Qwen Measure-by-Measure Batch Job
This job executes the Qwen model to generate beatmaps **line-by-line (beat-by-beat/measure-by-measure)**. This advanced generation process incorporates **Onsets**, **BPM**, and **Tempo** directly into its context window, allowing extremely detailed charts. This line-by-line feature is currently **exclusive to Qwen**.

To run this process, you **must be in the root directory** of the repository on the cluster:
```bash
ssh [EMAIL_ADDRESS]@hpc.ent.ohio.edu
cd /data/mg546924/llm_beatmap_generator
sbatch slurm_run_qwen_measure.sh
sbatch slurm_run_qwen_measure_all.sh # for all 20 songs
squeue -u $USER   # monitor job
```
Monitor its specific output using:
```bash
tail -f /data/mg546924/llm_beatmap_generator/logs/qwen_measure_<JOBID>.out
```

## 📊 Evaluation

Use `score_onset_detection.py` to compare AI-predicted onset timestamps against ground truth:
```bash
python score_onset_detection.py
```

Results are saved to `onset_score_summary_*.csv` with F1 score, precision, and recall metrics for each model.

---

## ⚙️ Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA A6000 (Ohio HPC) |
| Conda env | `deepresonance_env` |




## pushing files in the cluster

rsync -av /Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/pattern_finding_approach/fine name 
mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/pattern_finding_approach/
