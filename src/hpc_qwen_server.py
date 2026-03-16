"""
HPC Local Inference Server for Qwen2-Audio-7B-Instruct.

Loads the model from local disk (no internet/API key needed) and exposes
a simple HTTP API so the batch runner can send audio + prompt and get text back.

Start with:
    python src/hpc_qwen_server.py --model-dir /data/mg546924/models/Qwen2-Audio-7B-Instruct
"""

import os
import argparse
import base64
import tempfile
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

# ── Globals ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen2-Audio Beatmap Server")
_model = None
_processor = None

# ── Request / Response schemas ────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    audio_b64: str          # Base64-encoded raw audio bytes (wav/ogg/mp3)
    audio_filename: str     # Original filename (e.g. "Bad Ketchup.ogg") — used for extension
    prompt: str             # Full instruction prompt
    max_new_tokens: int = 8192

class GenerateResponse(BaseModel):
    text: str               # Raw model output (beatmap rows as text)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_dir: str):
    global _model, _processor
    print(f"Loading model from: {model_dir}")
    _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)
    _model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="cuda:0",   # pin to first GPU — avoids psutil/accelerate memory calc
        dtype=torch.float16,   # replaces deprecated torch_dtype
        trust_remote_code=True,
    )
    _model.eval()
    print("✅ Model loaded and ready.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

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
        # Load audio with soundfile (preferred for Qwen2-Audio processor)
        target_sr = _processor.feature_extractor.sampling_rate
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)

        # Build conversation — audio_url must point to the actual temp file
        conversation = [
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
            generated_ids = _model.generate(
                **inputs, 
                max_new_tokens=req.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05
            )

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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)
