import os
import sys
import torch
import torch.nn.functional as F
import librosa
import numpy as np

os.environ['HF_HOME'] = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints"

_processor = None
_model = None

def initialize_flamingo_model():
    """Load Music-Flamingo model and processor globally into VRAM."""
    global _processor, _model

    if _model is not None:
        return _model, _processor

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError as e:
        print(f"❌ Failed to import transformers classes: {e}")
        raise

    print("Loading Music-Flamingo Processor...", flush=True)
    _processor = AutoProcessor.from_pretrained(
        "nvidia/music-flamingo-hf", trust_remote_code=True
    )

    print("Loading Music-Flamingo Model (~30GB)...", flush=True)
    _model = AutoModelForCausalLM.from_pretrained(
        "nvidia/music-flamingo-hf",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    _model.eval()
    print("✅ Music-Flamingo loaded successfully.", flush=True)
    return _model, _processor


def get_flamingo_16_step_probabilities(audio_path, prompt, candidates,
                                        temperature=1.0,
                                        top_p=1.0,
                                        min_p=0.0,
                                        top_k=None,
                                        repetition_penalty=1.0,
                                        recent_history=None):
    """
    For each of the 16 candidate step patterns, compute the sequence log-probability
    via the model's cross-entropy loss on the candidate tokens, conditioned on the audio.
    Returns a probability dict sorted from highest to lowest.
    """
    global _processor, _model
    if _model is None:
        initialize_flamingo_model()

    # Load audio at Flamingo's preferred 16kHz sample rate
    y, sr = librosa.load(audio_path, sr=16000)

    cand_scores = []

    for cand in candidates:
        # Build the chat messages including candidate as assistant response
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "dummy"},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": cand}
        ]

        # Format the full conversation (prompt + candidate)
        full_text = _processor.apply_chat_template(messages, add_generation_prompt=False)
        # Format prompt-only (to know where to mask labels)
        prompt_only_messages = messages[:1]  # just user side
        prompt_text = _processor.apply_chat_template(prompt_only_messages, add_generation_prompt=True)

        # Tokenize full and prompt-only to find label boundary
        full_inputs = _processor(
            text=full_text,
            audio=[y],
            return_tensors="pt",
            sampling_rate=16000
        )
        prompt_inputs = _processor(
            text=prompt_text,
            audio=[y],
            return_tensors="pt",
            sampling_rate=16000
        )

        # Move to model device + dtype
        device = next(_model.parameters()).device
        full_inputs = {k: v.to(device) for k, v in full_inputs.items() if isinstance(v, torch.Tensor)}
        for k, v in full_inputs.items():
            if torch.is_floating_point(v):
                full_inputs[k] = v.to(torch.float16)

        # Build labels — mask prompt tokens with -100
        labels = full_inputs["input_ids"].clone().long()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[0, :prompt_len] = -100

        with torch.no_grad():
            outputs = _model(**full_inputs, labels=labels)
            loss = outputs.loss
            num_cand_tokens = (labels != -100).sum().item()
            if num_cand_tokens > 0:
                seq_log_prob = -loss.item() * num_cand_tokens
            else:
                seq_log_prob = -999.0  # fallback for empty candidates
            cand_scores.append(seq_log_prob)

    scores_tensor = torch.tensor(cand_scores, dtype=torch.float32)

    # ── Dynamic Repetition Penalty ──
    if recent_history and repetition_penalty > 1.0:
        for i, cand in enumerate(candidates):
            if cand == "0000" and recent_history.count("0000") < 4:
                continue  # Allow up to 3 rests freely

            count = recent_history.count(cand)
            if count > 0:
                dyn_penalty = 1.0 + (count * (repetition_penalty - 1.0))
                if cand == recent_history[-1]:
                    dyn_penalty *= 1.1
                scores_tensor[i] = scores_tensor[i] * dyn_penalty

    # ── Temperature Scaling ──
    if temperature != 1.0 and temperature > 0.0:
        scores_tensor = scores_tensor / temperature

    # ── Softmax → Probabilities ──
    probs = F.softmax(scores_tensor, dim=-1)

    # ── Min-P Filter ──
    if min_p > 0.0:
        probs[probs < min_p] = 0.0

    # ── Top-K Filter ──
    if top_k is not None and top_k > 0 and top_k < len(candidates):
        topk_vals, topk_indices = torch.topk(probs, top_k)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[topk_indices] = False
        probs[mask] = 0.0

    # ── Top-P Filter ──
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs[sorted_indices[sorted_indices_to_remove]] = 0.0

    # Renormalize
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.ones_like(probs) / len(candidates)

    probs_dict = {candidates[i]: probs[i].item() for i in range(len(candidates))}
    probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))

    return probs_dict
