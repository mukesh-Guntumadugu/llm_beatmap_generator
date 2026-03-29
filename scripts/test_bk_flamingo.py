#!/usr/bin/env python3
"""Music-Flamingo onset test on Bad Ketchup."""
import os, sys, csv, datetime, re, tempfile, gc, torch
import librosa, soundfile as sf

AUDIO   = os.environ.get("BENCHMARK_AUDIO",
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg")
OUT_DIR = os.environ.get("BENCHMARK_OUT",
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup")
PROJ    = os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")

os.environ["HF_HOME"] = os.path.join(PROJ, "Music-Flamingo", "checkpoints")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))
os.chdir(PROJ)

from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo

print("Loading Music-Flamingo model (~30GB)...", flush=True)
setup_music_flamingo()

print("Loading audio for chunking...", flush=True)
y, sr = librosa.load(AUDIO, sr=None)
duration = len(y) / sr
CHUNK = 2
all_onsets = []

print("Running chunked inference on Bad Ketchup...", flush=True)
for start in range(0, int(duration), CHUNK):
    end = min(start + CHUNK, duration)
    chunk = y[int(start*sr):int(end*sr)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, sr)
        tmp_path = tmp.name
    
    prompt = f"List all the onset timestamps in this {round(end-start,1)}s audio clip in milliseconds."
    response = generate_beatmap_with_flamingo(tmp_path, prompt)
    
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", response or "")
    for n in nums:
        # Shift onset by chunk start time
        all_onsets.append(int(round(float(n) + start*1000)))
    
    os.remove(tmp_path)
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Chunk {start:.0f}s-{end:.0f}s → {len(nums)} onsets (Raw: {str(response)[:80]}...)", flush=True)

ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = os.path.join(OUT_DIR, f"FlamingoChunked_TEST_Bad_Ketchup_{ts}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in all_onsets: w.writerow([ms])

print(f"✅ Music-Flamingo found {len(all_onsets)} onsets", flush=True)
print(f"   Saved to: {out}", flush=True)
