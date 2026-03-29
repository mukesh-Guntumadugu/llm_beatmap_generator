# 🎵 DeepResonance Chat Interface

## What is this?

`chat_deepresonance.py` is an interactive terminal chat interface for the **DeepResonance** AI model — a multimodal music understanding model built on top of LLaMA (Vicuna 7B). 

DeepResonance can **listen to audio files** and answer questions about them, such as:
- What genre is this music?
- What instruments are playing?
- Describe the tempo and rhythm.
- What musical events happen at the start?

> ⚠️ **Important:** DeepResonance is a music analysis model, NOT a general-purpose chatbot.  
> It always needs an audio file (or at least a dummy silence) to function — it was trained that way.  
> If you send text only, it will try to describe an imaginary audio instead of answering your question.

---

## Where does it run?

This runs on the **Ohio HPC cluster** (`hpc.ent.ohio.edu`), specifically on a GPU node with an NVIDIA A6000.  
It cannot run on a laptop because the model weights are ~12GB and require a powerful GPU.

---

## How to Run It (Step by Step)

### Step 1 — SSH into the cluster login node
```bash
ssh mg546924@hpc.ent.ohio.edu
```

### Step 2 — Request an interactive GPU session from Slurm
This gives you a live bash shell inside a real GPU node:
```bash
srun -p defq --gres=gpu:1 --pty bash
```
Wait a few seconds. When you see `bash-5.1$`, you are inside a GPU node.

### Step 3 — Launch the chat
Use the **full Python path** from the conda environment (do NOT use just `python`):
```bash
/data/mg546924/conda_envs/deepresonance_env/bin/python /data/mg546924/llm_beatmap_generator/chat_deepresonance.py
```

### Step 4 — Wait for the model to load
This takes about **60–90 seconds**. You will see it loading the visual encoder and language decoder:
```
Initializing visual encoder from ../ckpt/...
Visual encoder initialized.
Initializing language decoder from ../ckpt/...
Loading checkpoint shards: 100%|██████████| 2/2
Language decoder initialized.
[!] LLM initialized.

✅ DeepResonance Online! Type /exit to quit.
```

Once you see `✅ DeepResonance Online!`, you are ready to chat.

---

## How to Chat With It

### Option A — With an audio file (RECOMMENDED)
Paste the full path to any `.ogg`, `.mp3`, or `.wav` file on the cluster:

```
🎵 Audio file path (or Enter to skip): /data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg

🗣️  Your prompt: What instruments do you hear in this music?
```

### Option B — Without audio (text only)
Just press **Enter** with nothing typed when it asks for the audio file.  
The model will still respond, but it will not be analyzing any real audio — it may hallucinate a description.

```
🎵 Audio file path (or Enter to skip): [press Enter]

🗣️  Your prompt: What is a musical onset?
```

### Ending the session
Type `/exit` at any prompt to quit cleanly.

---

## Example Questions to Ask

| Prompt | What it does |
|--------|--------------|
| `What genre is this music?` | Classifies the audio genre |
| `What instruments are playing?` | Identifies instruments |
| `Describe the tempo and rhythm` | Analyzes beat and pace |
| `What musical events happen in the first 10 seconds?` | Time-based analysis |
| `Is this music energetic or calm?` | Mood/energy classification |
| `List all the onset timestamps in milliseconds` | Onset detection |

---

## Why Not Just Use Ollama or ChatGPT?

DeepResonance is a **specialized research model** — it was fine-tuned specifically on music datasets like MusicCaps, MusicQA, and GTZAN. Standard LLMs like ChatGPT cannot actually hear or process audio. DeepResonance can genuinely listen to the waveform and reason about it.

Ollama runs standard text models (LLaMA, Mistral, etc.) — it has no idea how to load DeepResonance's custom multimodal architecture.

`chat_deepresonance.py` is the purpose-built interface for this specific model.

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | You used `python` instead of the full conda path. Use `/data/mg546924/conda_envs/deepresonance_env/bin/python` |
| `No CUDA GPUs are available` | You forgot to `srun` and are on the login node. Run Step 2 first. |
| `IndexError: list index out of range` | Older version of the script. Re-download the latest `chat_deepresonance.py` |
| `libcusparse.so.11: cannot open shared object file` | This is a harmless warning from bitsandbytes — the model still runs fine. Ignore it. |
| Model loads but gives weird answers | Make sure you provide a real audio file path. Text-only mode makes the model hallucinate. |

---

## Files in This Project

| File | Purpose |
|------|---------|
| `chat_deepresonance.py` | **This file** — interactive chat interface |
| `extract_deepresonance_onsets.py` | Batch onset detection — runs all 20 songs automatically |
| `slurm_run_deepresonance.sh` | Slurm batch script to submit `extract_deepresonance_onsets.py` as a background job |
| `DeepResonance/` | The model code and weights |
| `src/musicForBeatmap/` | Your music library + generated CSV onset files |
