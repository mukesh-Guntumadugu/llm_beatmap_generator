#!/usr/bin/env python3
"""Librosa onset test on Bad Ketchup."""
import librosa, csv, datetime, os

AUDIO = os.environ.get("BENCHMARK_AUDIO", 
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg")
OUT_DIR = os.environ.get("BENCHMARK_OUT",
    "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup")

print("Loading audio...", flush=True)
y, sr = librosa.load(AUDIO, sr=None)
print(f"Audio loaded: {len(y)/sr:.1f}s at {sr}Hz", flush=True)

onsets = [round(float(t)*1000, 1) for t in librosa.onset.onset_detect(y=y, sr=sr, units='time')]

ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
out = os.path.join(OUT_DIR, f"Librosa_TEST_Bad_Ketchup_{ts}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset_ms"])
    for ms in onsets: w.writerow([ms])

print(f"✅ Librosa found {len(onsets)} onsets", flush=True)
print(f"   Saved to: {out}", flush=True)
