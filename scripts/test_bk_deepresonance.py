#!/usr/bin/env python3
"""DeepResonance onset test on Bad Ketchup — chunked 5s windows."""
import os, sys, gc, re, csv, datetime, tempfile, torch
import librosa, soundfile as sf

# Fix bitsandbytes/CUDA missing libcusparse.so.11
os.environ["LD_LIBRARY_PATH"] = f"/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

AUDIO   = os.environ.get("BENCHMARK_AUDIO",
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg")
OUT_DIR = os.environ.get("BENCHMARK_OUT",
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup")
PROJ    = os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")
CKPT    = os.path.join(PROJ, "DeepResonance", "ckpt")

sys.path.insert(0, os.path.join(PROJ, "DeepResonance", "code"))
os.chdir(os.path.join(PROJ, "DeepResonance", "code"))

from inference_deepresonance import DeepResonancePredict

args = {
    "stage": 2, "mode": "test", "dataset": "musiccaps",
    "project_path": os.path.join(PROJ, "DeepResonance", "code"),
    "llm_path": os.path.join(CKPT, "pretrained_ckpt", "vicuna_ckpt", "7b_v0"),
    "imagebind_path": os.path.join(CKPT, "pretrained_ckpt", "imagebind_ckpt", "huge"),
    "imagebind_version": "huge",
    "max_length": 512, "max_output_length": 512,
    "num_clip_tokens": 77, "gen_emb_dim": 768,
    "preencoding_dropout": 0.1, "num_preencoding_layers": 1,
    "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False,
    "prompt": "", "prellmfusion": True, "prellmfusion_dropout": 0.1,
    "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.001,
    "ckpt_path": os.path.join(CKPT, "DeepResonance_data_models", "ckpt",
                              "deepresonance_beta_delta_ckpt", "delta_ckpt",
                              "deepresonance", "7b_tiva_v0"),
}

print("Loading DeepResonance model...", flush=True)
model = DeepResonancePredict(args)

print("Loading audio...", flush=True)
y, sr = librosa.load(AUDIO, sr=None)
duration = len(y) / sr
CHUNK = 2
all_onsets = []

for start in range(0, int(duration), CHUNK):
    end = min(start + CHUNK, duration)
    chunk = y[int(start*sr):int(end*sr)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, sr)
        tmp_path = tmp.name
    inputs = {
        "inputs": ["<Audio>"],
        "instructions": [f"List all onset timestamps in milliseconds for this {round(end-start,1)}s clip."],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]],
        "mm_root_path": os.path.dirname(tmp_path),
        "outputs": [""],
    }
    resp = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.001, stops_id=[[835]])
    if isinstance(resp, list): resp = resp[0]
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", resp or "")
    for n in nums:
        all_onsets.append(int(round(float(n) + start*1000)))
    os.remove(tmp_path)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Chunk {start:.0f}s-{end:.0f}s → {len(nums)} onsets", flush=True)

ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = os.path.join(OUT_DIR, f"DeepResonance_TEST_Bad_Ketchup_{ts}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in all_onsets: w.writerow([ms])

print(f"✅ DeepResonance found {len(all_onsets)} onsets", flush=True)
print(f"   Saved to: {out}", flush=True)
