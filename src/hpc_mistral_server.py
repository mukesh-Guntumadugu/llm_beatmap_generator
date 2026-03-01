"""
HPC Local Inference Server for Mistral-7B-Instruct.

Loads Mistral from local disk (no internet/API key needed) and exposes
an HTTP API compatible with hpc_qwen_server.py so the batch runner can
send audio + prompt and receive a beatmap JSON response.

Because Mistral is text-only, audio is decoded on the server and
converted to rich musical features (BPM, onsets, beats, RMS energy)
before being fed to the model as a structured text prompt.

Start with:
    python src/hpc_mistral_server.py \
        --model-dir /data/mg546924/models/Mistral-7B-Instruct-v0.3

Download model (run once on login node with internet access):
    python -c "
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.3',
        cache_dir='/data/mg546924/models/Mistral-7B-Instruct-v0.3'
    )
    AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.3',
        cache_dir='/data/mg546924/models/Mistral-7B-Instruct-v0.3'
    )
    "
"""

import argparse
import base64
import json
import os
import tempfile

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Globals ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Mistral-7B Beatmap Server")
_model = None
_tokenizer = None

# ── Request / Response schemas ────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    audio_b64: str           # Base64-encoded audio bytes (wav/ogg/mp3)
    audio_filename: str      # Original filename — used for file extension
    difficulty: str = "Hard" # Easy / Medium / Hard / Expert
    max_new_tokens: int = 8192


class GenerateResponse(BaseModel):
    text: str                # JSON array of beatmap row objects


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_dir: str):
    global _model, _tokenizer
    print(f"Loading tokenizer from: {model_dir}")
    _tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"Loading model from: {model_dir}  (float16, cuda:0)")
    _model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    _model.eval()
    print("✅ Mistral model loaded and ready.")


# ── Audio feature extraction ──────────────────────────────────────────────────
def _extract_features(audio_path: str) -> dict:
    """
    Extract BPM, onset/beat times, and per-16th-note RMS energy from audio.
    Returns a dict ready to embed in a text prompt.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM + beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # RMS energy per 16th-note slot
    spb = 60.0 / max(bpm, 1)
    slot_dur = spb / 4
    n_slots = max(1, int(duration / slot_dur))
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    energy: list[float] = []
    for s in range(n_slots):
        t0, t1 = s * slot_dur, (s + 1) * slot_dur
        mask = (rms_times >= t0) & (rms_times < t1)
        val = float(np.mean(rms[mask])) if mask.any() else 0.0
        energy.append(round(val, 4))

    return {
        "bpm": round(bpm, 2),
        "duration": round(duration, 3),
        "onset_times": [round(t, 3) for t in onset_times[:80]],
        "beat_times": [round(t, 3) for t in beat_times[:80]],
        "energy_by_16th": energy[:128],
    }


# ── Prompt builder ────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a StepMania DDR beatmap generator. You receive musical analysis "
    "data (BPM, onset times, beat times, RMS energy per 16th-note slot) and "
    "output ONLY a JSON array of step objects with these fields:\n"
    "  time_ms (float), beat_position (float), notes (str 4-char or ','),\n"
    "  placement_type (int), note_type (int), confidence (float), instrument (str)\n\n"
    "RULES:\n"
    "• Each measure ends with notes=',' (placement_type=-1, note_type=-1).\n"
    "• Each measure has EXACTLY 4, 8, or 16 note rows. 16=sixteenth (preferred).\n"
    "• Fill silent slots with notes='0000'.\n"
    "• Cover the ENTIRE duration — do NOT stop early.\n"
    "• Output ONLY valid JSON. No markdown, no explanation."
)


def _build_prompt(features: dict, difficulty: str) -> str:
    return (
        f"Difficulty: {difficulty}\n"
        f"Duration: {features['duration']:.3f}s\n"
        f"BPM: {features['bpm']}\n"
        f"Onsets (s): {json.dumps(features['onset_times'])}\n"
        f"Beats  (s): {json.dumps(features['beat_times'])}\n"
        f"Energy/16th: {json.dumps(features['energy_by_16th'])}\n\n"
        f"Generate the full {difficulty} beatmap JSON array."
    )


# ── Inference ─────────────────────────────────────────────────────────────────
def _run_inference(prompt: str, max_new_tokens: int) -> str:
    """Run Mistral chat inference and return the assistant reply."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    # apply_chat_template handles [INST] / [/INST] formatting
    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic, lower hallucination
            temperature=1.0,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Strip prompt tokens
    new_ids = output_ids[:, input_ids.shape[1]:]
    return _tokenizer.decode(new_ids[0], skip_special_tokens=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Decode audio → temp file
    audio_bytes = base64.b64decode(req.audio_b64)
    suffix = os.path.splitext(req.audio_filename)[-1] or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Extract features
        features = _extract_features(tmp_path)
        prompt   = _build_prompt(features, req.difficulty)

        # Run model
        response_text = _run_inference(prompt, req.max_new_tokens)
        return GenerateResponse(text=response_text)

    finally:
        os.remove(tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/data/mg546924/models/Mistral-7B-Instruct-v0.3",
        help="Path to Mistral model directory on the cluster.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    load_model(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)
