"""
HPC Local Inference Server for Qwen2-Audio-7B-Instruct.

Loads the model from local disk (no internet/API key needed) and exposes
a simple HTTP API so the batch runner can send audio + prompt and get text back.

Uses Outlines constrained decoding — the model physically cannot output invalid
tokens. Output is guaranteed to match the BeatCSV schema (same guarantee as
Gemini's response_schema).

Start with:
    python src/hpc_qwen_server.py --model-dir /data/mg546924/models/Qwen2-Audio-7B-Instruct
"""

import os
import json
import argparse
import base64
import tempfile
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

# ── Globals ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen2-Audio Beatmap Server")
_model = None
_processor = None
_prefix_fn = None   # lm-format-enforcer constrained decoding prefix function

# ── Beatmap output schema (mirrors Gemini's BeatCSV exactly) ─────────────────
class BeatCSV(BaseModel):
    time_ms: float = Field(..., description="Exact timestamp in milliseconds")
    beat_position: float = Field(..., description="Beat number from song start")
    notes: str = Field(..., description="4-char StepMania row e.g. '1000' or ',' for measure end")
    placement_type: int = Field(..., description="0=unsure,1=onset,2=beat,3=grid,4=percussive,5=unaligned,-1=separator")
    note_type: int = Field(..., description="0=whole,1=half,2=quarter,3=eighth,4=extended,-1=separator")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    instrument: str = Field(..., description="kick/snare/bass/melody/guitar/synth/unknown/separator")

class BeatmapOutput(BaseModel):
    rows: List[BeatCSV]

# ── Request / Response schemas ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    audio_b64: str          # Base64-encoded raw audio bytes (wav/ogg/mp3)
    audio_filename: str     # Original filename (e.g. "Bad Ketchup.ogg") — used for extension
    system_prompt: str      # Static system instruction (CSV rules, format spec)
    prompt: str             # Dynamic user turn (difficulty, duration, BPM, onsets)
    max_new_tokens: int = 16384
    chunk_duration_sec: float = 20.0  # Audio chunk duration in seconds

class GenerateResponse(BaseModel):
    text: str               # Raw model output (beatmap rows as text)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_dir: str, lora_dir: str = None):
    global _model, _processor, _prefix_fn
    print(f"Loading model from: {model_dir}")
    _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)
    _model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="cuda:0",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    
    if lora_dir:
        print(f"Applying LoRA adapter from: {lora_dir}")
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, lora_dir)
        
    _model.eval()

    # ── Set up lm-format-enforcer constrained decoding ───────────────────────
    # Same guarantee as Gemini's response_schema: model cannot emit invalid tokens.
    # We now enforce raw CSV rows to match the prompt.
    try:
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn
        )
        
        # Regex for CSV rows: time_ms,beat_pos,notes,placement,note_type,conf,instrument
        # Allows notes like '1000' or '","' for separators
        csv_row = r"\d+\.\d+,\d+\.\d+,(?:[0123M]{4}|\",\"),-?\d+,-?\d+,\d+\.\d+,[a-z]+"
        csv_regex = f"({csv_row}\n)+"
        schema_parser = RegexParser(csv_regex)
        
        _prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            _processor.tokenizer, schema_parser
        )
        print("✅ lm-format-enforcer constrained decoding enabled (Regex: CSV Rows).")
    except ImportError:
        _prefix_fn = None
        print("⚠️  lm-format-enforcer not installed — falling back to unconstrained generation.")
        print("   To enable: pip install lm-format-enforcer")
    except Exception as e:
        _prefix_fn = None
        print(f"⚠️  Constrained decoding setup failed: {e} — using unconstrained generation.")

    print("✅ Model loaded and ready.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "constrained_decoding": _prefix_fn is not None,
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Decode audio bytes and write to a temp file
    audio_bytes = base64.b64decode(req.audio_b64)
    suffix = os.path.splitext(req.audio_filename)[-1] or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Load audio
        target_sr = _processor.feature_extractor.sampling_rate
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)

        # Build conversation — system role = CSV rules, user role = audio + per-song info
        conversation = [
            {
                "role": "system",
                "content": req.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": f"file://{tmp_path}"},
                    {"type": "text",  "text": req.prompt},
                ],
            }
        ]
        text = _processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Pass audio array + sampling_rate explicitly
        inputs = _processor(
            text=text,
            audio=[y],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ).to(_model.device)

        with torch.no_grad():
            # min_new_tokens: ~4 rows/sec × 5 tokens/row is a reasonable floor
            min_tokens = max(32, int(req.chunk_duration_sec * 4 * 5))
            
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                min_new_tokens=min_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            
            # Plug in lm-format-enforcer prefix function if available
            # This restricts token choices so the model CANNOT output invalid JSON
            if _prefix_fn is not None:
                generate_kwargs["prefix_allowed_tokens_fn"] = _prefix_fn
            
            generated_ids = _model.generate(**generate_kwargs)

        # Strip the input tokens from the output
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response_text = _processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return GenerateResponse(text=response_text)

    finally:
        os.remove(tmp_path)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/data/mg546924/models/Qwen2-Audio-7B-Instruct",
        help="Path to the downloaded model directory on the cluster."
    )
    parser.add_argument(
        "--lora-dir",
        default=None,
        help="Path to the trained LoRA adapter directory on the cluster."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model_dir, args.lora_dir)
    uvicorn.run(app, host=args.host, port=args.port)
