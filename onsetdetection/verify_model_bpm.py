"""
verify_model_bpm.py
===================
Batch evaluates the unified mathematical BPM of an audio dataset against zero-shot LLMs.

Usage:
    python3 onsetdetection/verify_model_bpm.py --batch_dir <path> --model qwen
"""

import argparse
import sys
import os
import csv
import torch
import librosa
import numpy as np
import tempfile
import soundfile as sf
import gc

ROOT = "/data/mg546924/llm_beatmap_generator"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

PROMPT = "Listen to this audio clip. What is the unified global BPM (Beats Per Minute) of this song? Output only the exact integer number."
SEP = "─" * 68

def find_audio_file(folder_path: str) -> str:
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._"):
            return os.path.join(folder_path, f)
    return None

def find_sm_file(folder_path: str) -> str:
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.sm', '.ssc')) and not f.startswith("._"):
            return os.path.join(folder_path, f)
    return None

def extract_human_bpm_from_sm(sm_path: str) -> str:
    if not sm_path or not os.path.exists(sm_path):
        return "N/A"
    try:
        import re
        with open(sm_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'#BPMS:.*?=([\d\.]+)', content)
        if match:
            return str(round(float(match.group(1)), 2))
    except Exception:
        pass
    return "N/A"

# ── Math Ground Truth ─────────────────────────────────────────────────────────

def run_librosa_batch(song_dirs: list, out_csv: str):
    print(f"Running Librosa & Human Ground Truth extraction across {len(song_dirs)} songs...")
    results = [["Song_Name", "Human_Ground_Truth_BPM", "Librosa_Global_BPM", "Librosa_Min_Drift", "Librosa_Max_Drift"]]
    
    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        sm_path = find_sm_file(song_dir)
        
        if not audio_path: continue
        
        print(f" [{idx+1}/{len(song_dirs)}] {song_name}")
        y, sr = librosa.load(audio_path, sr=None)
        
        # Human
        human_bpm = extract_human_bpm_from_sm(sm_path)
        
        # Global Math
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        val = tempo[0] if isinstance(tempo, np.ndarray) else float(tempo)
        global_bpm = round(float(val), 2)
        
        # Drift bounds
        dyn, _ = librosa.beat.beat_track(y=y, sr=sr, aggregate=None)
        min_t, max_t = float(np.min(dyn)), float(np.max(dyn)) if len(dyn)>0 else (0.0, 0.0)
        
        results.append([song_name, human_bpm, global_bpm, round(min_t, 2), round(max_t, 2)])
        
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(results)
    print(f"Saved: {out_csv}")


# ── MuMu-LLaMA ────────────────────────────────────────────────────────────────
def run_mumu_batch(song_dirs: list, out_csv: str):
    from mumu_measure_interface import initialize_mumu_model
    import llama
    print("Loading MuMu-LLaMA...")
    model, _ = initialize_mumu_model()
    formatted = llama.utils.format_prompt(PROMPT)
    
    results = [["Song_Name", "MuMu_BPM_Estimate"]]
    
    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        if not audio_path: continue
        
        print(f" [{idx+1}/{len(song_dirs)}] {song_name}")
        y, sr = librosa.load(audio_path, sr=24000, duration=30.0)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = model.generate(prompts=[formatted], audios=y, max_gen_len=16, temperature=0.1)
        
        ans = str(out[0]).strip() if isinstance(out, list) else str(out).strip()
        results.append([song_name, ans])
        
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(results)
    print(f"Saved: {out_csv}")

# ── Qwen2-Audio ───────────────────────────────────────────────────────────────
def run_qwen_batch(song_dirs: list, out_csv: str):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    print("Loading Qwen2-Audio...")
    model_path = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    
    results = [["Song_Name", "Qwen_BPM_Estimate"]]
    
    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        if not audio_path: continue
        
        print(f" [{idx+1}/{len(song_dirs)}] {song_name}")
        y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate, duration=30.0)
        
        messages = [
            {"role": "system", "content": "You are a concise musical assistant."},
            {"role": "user", "content": [{"type": "audio", "audio_url": "dummy"}, {"type": "text", "text": PROMPT}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[y], sampling_rate=sr, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, temperature=0.1, do_sample=False, use_cache=False)
        ans = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        results.append([song_name, ans])
        
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(results)
    print(f"Saved: {out_csv}")

# ── DeepResonance ─────────────────────────────────────────────────────────────
def run_deepresonance_batch(song_dirs: list, out_csv: str):
    from deepresonance_measure_interface import initialize_deepresonance_model
    print("Loading DeepResonance...")
    model = initialize_deepresonance_model()
    results = [["Song_Name", "DeepResonance_BPM_Estimate"]]
    
    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        if not audio_path: continue
        
        print(f" [{idx+1}/{len(song_dirs)}] {song_name}")
        y, sr = librosa.load(audio_path, sr=24000, duration=30.0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            tmp_path = tmp.name
            
        inputs = {
            "inputs": ["<Audio>"],
            "instructions": [PROMPT],
            "mm_names": [["audio"]],
            "mm_paths": [[os.path.basename(tmp_path)]],
            "mm_root_path": os.path.dirname(tmp_path),
            "outputs": [""],
        }
        
        try:
            resp = model.predict(inputs, max_tgt_len=16, top_p=1.0, temperature=0.1, stops_id=[[835]])
            ans = str(resp[0]).strip() if isinstance(resp, list) and resp else str(resp).strip()
            results.append([song_name, ans])
        except Exception as e:
            results.append([song_name, f"ERROR: {e}"])
        finally:
            os.unlink(tmp_path)
            
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(results)
    print(f"Saved: {out_csv}")


# ── Music-Flamingo ────────────────────────────────────────────────────────────
def run_flamingo_batch(song_dirs: list, out_csv: str):
    from music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
    print("Loading Music-Flamingo...")
    setup_music_flamingo()
    results = [["Song_Name", "Flamingo_BPM_Estimate"]]
    
    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        if not audio_path: continue
        
        print(f" [{idx+1}/{len(song_dirs)}] {song_name}")
        y, sr = librosa.load(audio_path, sr=24000, duration=30.0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            tmp_path = tmp.name

        fl_prompt = f"<Audio>\n{PROMPT}"
        try:
            ans = generate_beatmap_with_flamingo(tmp_path, fl_prompt)
            results.append([song_name, str(ans).strip()])
        finally:
            os.unlink(tmp_path)
            
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(results)
    print(f"Saved: {out_csv}")

# ── Main Engine ────────────────────────────────────────────────────────────────

MODEL_ROUTES = {
    "librosa":       run_librosa_batch,
    "mumu":          run_mumu_batch,
    "qwen":          run_qwen_batch,
    "deepresonance": run_deepresonance_batch,
    "flamingo":      run_flamingo_batch,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True, help="Path to Fraxtil dataset dir")
    parser.add_argument("--model", choices=list(MODEL_ROUTES.keys()), required=True)
    parser.add_argument("--timestamp", default="", help="Optional string timestamp to append to filenames")
    args = parser.parse_args()
    
    if not os.path.exists(args.batch_dir):
        print(f"Dataset dir not found: {args.batch_dir}")
        return

    # Convert --batch_dir to absolute immediately so model imports don't break relative paths when they chdir internally!
    abs_batch_dir = os.path.abspath(args.batch_dir)

    print("Spidering dataset folders recursively...", flush=True)
    
    song_dirs = []
    # Force recursive Spidering for jagged nested Beatmap Packs
    for root_dir, list_dirs, list_files in os.walk(abs_batch_dir, followlinks=True):
        # Quickly skip macOS junk metadata
        if os.path.basename(root_dir).startswith("_"):
            continue
            
        # Is there at least one valid audio file explicitly in this sub-folder?
        has_audio = any(f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._") for f in list_files)
        if has_audio:
            song_dirs.append(root_dir)

    song_dirs = sorted(list(set(song_dirs)))

    print("\n" + SEP)
    print(f"🎵  BATCH BPM VERIFICATION: {args.model.upper()}")
    print(SEP + "\n")

    timestamp_suffix = f"_{args.timestamp}" if args.timestamp else ""
    out_csv = os.path.abspath(os.path.join("onsetdetection", f"BPM_Estimates_{args.model.upper()}{timestamp_suffix}.csv"))
    exec_func = MODEL_ROUTES[args.model]
    exec_func(song_dirs, out_csv)
    
if __name__ == "__main__":
    main()
