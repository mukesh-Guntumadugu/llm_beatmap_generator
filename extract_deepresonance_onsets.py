"""
extract_deepresonance_onsets.py
========================
Sends each of the 20 Fraxtil songs to the local DeepResonance model and asks it to
predict audio onset times in milliseconds. Saves results to CSV files.

This mimics the exact framework, prompt, and output parsing system of extract_gemini_onsets.py,
but relies on the local multimodal audio inference rather than the Gemini API.
"""

import os
import re
import csv
import sys
import json
import time
import datetime
import argparse
import librosa
import soundfile as sf
import numpy as np
from typing import Optional, List
import torch

# --- DEEPSPEED C++ COMPILER BYPASS ---
# DeepSpeed 0.9.3 demands triton.ops, but Slurm lacks the NVCC compiler to upgrade DeepSpeed.
# We inject a MagicMock here to completely fake out the entire DeepSpeed triton module tree!
import sys
from unittest.mock import MagicMock
try:
    import triton
except ImportError:
    sys.modules['triton'] = MagicMock()
sys.modules['triton.ops'] = MagicMock()
sys.modules['triton.ops.matmul_perf_model'] = MagicMock()
# -------------------------------------

# ── Import DeepResonance ──────────────────────────────────────────────────────
# Add DeepResonance code/ to the Python path
DEEPRESONANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepResonance", "code")
sys.path.append(DEEPRESONANCE_DIR)

try:
    # Requires deepresonance conda env installed correctly
    from inference_deepresonance import DeepResonancePredict
except ImportError as e:
    print(f"⚠️  Could not import DeepResonance: {e}")
    print("Ensure you are running within deepresonance_env conda environment natively on the A6000!")


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

def build_onset_prompt(duration_sec: float) -> str:
    prompt = (
        f"<Audio>\nPlease listen to the audio and identify all musical onset events (transients) in this {round(duration_sec, 2)}-second clip. "
        "Output a continuous single-line comma-separated array of all your found timestamps in strictly absolute milliseconds."
    )
    return prompt


def find_audio_file(song_dir: str) -> Optional[str]:
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None


def parse_onsets_from_response(response_text: str) -> List[float]:
    text = response_text.strip()
    text_clean = re.sub(r"```[\w]*\n?", "", text).strip()
    onsets = []
    
    # Try array parsing
    try:
        match = re.search(r'\[([\d.,\s]+)\]', text_clean)
        if match:
            arr = json.loads('[' + match.group(1) + ']')
            for val in arr:
                ms = float(val)
                if 0.0 <= ms <= 600_000:
                    onsets.append(round(ms, 2))
            if onsets:
                return sorted(set(onsets))
    except Exception:
        pass
        
    # Fallback numerical search
    numbers = re.findall(r'\b(\d+(?:[.,]\d+)?)\b', text_clean)
    for n in numbers:
        n = n.replace(',', '.')
        try:
            ms = float(n)
            if 0.0 <= ms <= 600_000:
                onsets.append(round(ms, 2))
        except ValueError:
            pass
    return sorted(set(onsets))


def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str, chunk_sec: int) -> str:
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"DeepResonance_onsets_{safe_name}_{timestamp}_{chunk_sec}sec.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_ms"])
        for ms in onset_ms:
            writer.writerow([ms])
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Extract onsets using local DeepResonance.")
    parser.add_argument("--ckpt_path", type=str, default="../DeepResonance/ckpt",
                        help="Path to the DeepResonance model directory")
    args = parser.parse_args()

    # Model Init
    print(f"Loading DeepResonance Model from {args.ckpt_path}...")
    try:
        model_args = {
            # Run-time args
            'model': 'deepresonance',
            'mode': 'test',
            'dataset': 'musiccaps',
            'server': 'local',
            'result_file_name': '',
            'stage': 2,
            'topp': 1.0,
            'temp': 0.1,
            # From base.yaml
            'pretrained_ckpt_path': os.path.join(args.ckpt_path, 'pretrained_ckpt'),
            'vicuna_version': '7b_v0',
            'imagebind_version': 'huge',
            'max_length': 512,
            'max_output_length': 512,
            'num_clip_tokens': 77,
            'gen_emb_dim': 768,
            'preencoding_dropout': 0.1,
            'num_preencoding_layers': 1,
            # From stage_2.yaml
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'freeze_lm': False,
            'freeze_input_proj': False,
            'freeze_output_proj': False,
            'prompt': '',
            # prellmfusion
            'prellmfusion': True,
            'prellmfusion_dropout': 0.1,
            'num_prellmfusion_layers': 1,
            'imagebind_embs_seq': True,
            # ckpt
            'ckpt_path': os.path.join(args.ckpt_path, 'DeepResonance_data_models', 'ckpt', 'deepresonance_beta_delta_ckpt', 'delta_ckpt', 'deepresonance', '7b_tiva_v0'),
        }
        dr_predictor = DeepResonancePredict(model_args)
    except Exception as e:
        print(f"Failed to load model logic. Once full weights are downloaded, this will run: {e}")
        return

    song_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_")])
    total_songs = 0

    print("─" * 110)
    for i, song_name in enumerate(song_dirs):
        song_dir  = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        audio_path = find_audio_file(song_dir)
        if not audio_path: continue
        
        try:
            # Load full audio into memory once
            y, sr = librosa.load(audio_path, sr=None)
            
            CHUNK_SIZES = [1, 5, 10, 15, 20]
            for chunk_sec in CHUNK_SIZES:
                print(f"  [{i+1}/{len(song_dirs)}] Inferring: {song_name} ({chunk_sec}sec chunks)...", flush=True)
                chunk_samples = int(chunk_sec * sr)
                num_chunks = int(np.ceil(len(y) / chunk_samples))
                
                all_onsets_ms = []
                
                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * chunk_samples
                    end_sample = min((chunk_idx + 1) * chunk_samples, len(y))
                    chunk_y = y[start_sample:end_sample]
                    
                    # Create temporary audio slice
                    tmp_audio_path = f"/tmp/DR_temp_{chunk_sec}s_chunk{chunk_idx}.wav"
                    sf.write(tmp_audio_path, chunk_y, sr)
                    
                    chunk_duration = len(chunk_y) / sr
                    prompt = build_onset_prompt(chunk_duration)
                    
                    # Target temporal audio path for DeepResonance
                    inputs = {
                        "inputs": [prompt],
                        "instructions": [prompt],
                        "mm_names": [["audio"]],
                        "mm_paths": [[os.path.basename(tmp_audio_path)]],
                        "mm_root_path": os.path.dirname(tmp_audio_path),
                        "outputs": [""],
                    }
                    
                    try:
                        response_text = dr_predictor.predict(
                            inputs,
                            max_tgt_len=512,
                            top_p=1.0,
                            temperature=0.1,
                            stops_id=[[835]],
                        )
                        
                        if isinstance(response_text, list):
                            response_text = response_text[0].split("\n###")[0]
                        
                        # Parse predicted milliseconds
                        parsed_ms = parse_onsets_from_response(response_text)
                        
                        # Shift predictions to match actual timeline
                        chunk_start_ms = chunk_idx * chunk_sec * 1000
                        shifted_ms = [round(ms + chunk_start_ms, 2) for ms in parsed_ms]
                        all_onsets_ms.extend(shifted_ms)
                        
                    except Exception as e:
                        print(f"    ❌ Error on chunk {chunk_idx}: {e}")
                    finally:
                        # Ensure temp file is scrubbed immediately
                        if os.path.exists(tmp_audio_path):
                            os.remove(tmp_audio_path)
                
                # Deduplicate, sort, and save final CSV 
                all_onsets_ms = sorted(set(all_onsets_ms))
                out_path = save_onsets_csv(all_onsets_ms, song_name, song_dir, chunk_sec)
                print(f"    ✅ Extracted {len(all_onsets_ms)} total onsets for {chunk_sec}s windows!")
                
                # Full GC after each chunk_sec pass
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            total_songs += 1
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Error processing {song_name}: {e}")

    print(f"\n✅ Completed DeepResonance Batch! {total_songs}/{len(song_dirs)} processed.")

if __name__ == "__main__":
    main()
