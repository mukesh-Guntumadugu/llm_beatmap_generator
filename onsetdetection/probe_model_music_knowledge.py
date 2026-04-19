"""
probe_model_music_knowledge.py
==============================
Sends 3 basic music questions (NO audio) to each model to verify:
  1. Model is connected and reachable
  2. Model understands musical vocabulary (onset, beat, tempo)

Usage:
    python3 onsetdetection/probe_model_music_knowledge.py --model mumu
    python3 onsetdetection/probe_model_music_knowledge.py --model qwen
    python3 onsetdetection/probe_model_music_knowledge.py --model gemini
    python3 onsetdetection/probe_model_music_knowledge.py --model deepresonance
    python3 onsetdetection/probe_model_music_knowledge.py --model all
"""

import argparse
import sys
import os
import numpy as np

ROOT = "/data/mg546924/llm_beatmap_generator"
sys.path.insert(0, ROOT)

QUESTIONS = [
    "What is a musical onset? Give a one-sentence definition.",
    "If a song has a tempo of 120 BPM, how many beats occur in 10 seconds?",
    "Name three concrete ways to detect note onsets in an audio signal.",
]

SEP = "─" * 68

# ── MuMu-LLaMA ────────────────────────────────────────────────────────────────
def ask_mumu(question: str) -> str:
    from src.mumu_measure_interface import initialize_mumu_model
    import llama, torch

    model, _ = initialize_mumu_model()
    formatted  = llama.utils.format_prompt(question)

    # Pass 1 second of silence — neutral audio so MERT doesn't crash
    silent = np.zeros(24000, dtype=np.float32)

    with torch.no_grad():
        results = model.generate(
            prompts=[formatted],
            audios=silent,
            max_gen_len=256,
            temperature=0.2,
            top_p=0.9,
        )
    return results[0] if isinstance(results, list) else str(results)


# ── Qwen2-Audio ───────────────────────────────────────────────────────────────
def ask_qwen(question: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    inputs = tok(question, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=True)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Gemini ────────────────────────────────────────────────────────────────────
def ask_gemini(question: str) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(question).text


# ── DeepResonance ─────────────────────────────────────────────────────────────
def ask_deepresonance(question: str) -> str:
    """
    DeepResonance is audio-only, so we ask it to describe
    what it "hears" in a 1-second silent clip and embed the question
    as a text prompt alongside it.
    """
    import torch
    try:
        from src.deepresonance_interface import generate_with_deepresonance
        # Pass the question as caption/prompt; use silent audio
        result = generate_with_deepresonance(
            audio_array=np.zeros(24000, dtype=np.float32),
            sample_rate=24000,
            prompt=question,
        )
        return str(result)
    except Exception as e:
        return f"[DeepResonance probe error: {e}]"


# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_FUNCS = {
    "mumu":          ("MuMu-LLaMA",       ask_mumu),
    "qwen":          ("Qwen2-Audio",       ask_qwen),
    "gemini":        ("Gemini 1.5 Flash",  ask_gemini),
    "deepresonance": ("DeepResonance",     ask_deepresonance),
}


# ── Probe runner ───────────────────────────────────────────────────────────────
def probe_model(label: str, ask_fn):
    print(f"\n{'='*68}")
    print(f"  MODEL: {label}")
    print(f"{'='*68}")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n  Q{i}: {q}")
        print(f"  {SEP}")
        try:
            answer = ask_fn(q)
            # Indent each line of the answer for readability
            for line in str(answer).strip().splitlines():
                print(f"      {line}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Probe model music knowledge")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FUNCS.keys()) + ["all"],
        default="all",
        help="Which model to probe",
    )
    args = parser.parse_args()

    targets = (
        list(MODEL_FUNCS.items())
        if args.model == "all"
        else [(args.model, MODEL_FUNCS[args.model])]
    )

    print("\n🎵  Music Knowledge Probe")
    print(f"    Questions : {len(QUESTIONS)}")
    print(f"    Models    : {[t[0] for t in targets]}")

    for name, (label, fn) in targets:
        probe_model(label, fn)

    print("\n✅  Probe complete.\n")


if __name__ == "__main__":
    main()
