#!/usr/bin/env python3
import os
import sys
import tempfile
import gc
import csv
import random
import datetime
import time
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# DeepResonance needs LD_LIBRARY_PATH fix before anything else
os.environ["LD_LIBRARY_PATH"] = (
    "/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

PROJ = os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))

try:
    from src.deepresonance_measure_interface import initialize_deepresonance_model, get_deepresonance_16_step_probabilities
except ImportError:
    print("Could not import src.deepresonance_measure_interface. Ensure BENCHMARK_PROJ is correct.")
    sys.exit(1)

AUDIO_PATH = os.environ.get("BENCHMARK_AUDIO", "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg")
OUT_DIR = os.environ.get("BENCHMARK_OUT", os.path.dirname(AUDIO_PATH))

VALID_STEP_COMBOS = [
    "0001", "0010", "0100", "1000",
    "0011", "0101", "0110", "1001", "1010", "1100",
    "0111", "1011", "1101", "1110",
    "1111"
]

def main():
    print("Loading DeepResonance model...")
    initialize_deepresonance_model(PROJ)

    print("Loading audio and analyzing onsets via Librosa...")
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    global_bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    print(f"Global BPM detected: {global_bpm:.1f}")

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    beats_per_measure = 4
    beat_duration_sec = 60.0 / global_bpm
    measure_duration_sec = beat_duration_sec * beats_per_measure
    total_measures = int(np.ceil(duration / measure_duration_sec))

    print(f"Total Song Duration: {duration:.2f}s")
    print(f"Measure Duration: {measure_duration_sec:.3f}s")
    print(f"Total Measures: {total_measures}")
    print("--------------------------------------------------")

    global OUT_DIR
    OUT_DIR = os.path.join(OUT_DIR, "DeepResonance", "Wonsets_Wtempo_WBPM")
    os.makedirs(OUT_DIR, exist_ok=True)

    duration_min = duration / 60.0
    ALL_16_COMBOS = ["0000"] + VALID_STEP_COMBOS

    history_log = []
    global_linear_history = []
    txt_rows = []
    csv_rows = []

    for M in tqdm(range(total_measures), desc="Generating Beats (Beat-by-Beat)"):
        start_time = M * measure_duration_sec
        end_time = start_time + measure_duration_sec

        if start_time >= duration:
            break

        txt_rows.append(",")

        chunk_y = y[int(start_time*sr):int(min(end_time, duration)*sr)]
        target_len = int(measure_duration_sec * sr)
        if len(chunk_y) < target_len:
            chunk_y = np.pad(chunk_y, (0, target_len - len(chunk_y)), 'constant')

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk_y, sr)
            tmp_path = tmp.name

        m_onsets = [round(t, 2) for t in onset_times if start_time <= t < end_time]
        measure_step_outputs = []

        for b in range(4):
            beat_time = start_time + b * beat_duration_sec

            onsets_in_slot = [t for t in m_onsets if beat_time <= t < beat_time + beat_duration_sec]
            has_onset = len(onsets_in_slot) > 0
            onsets_timestamps_str = "|".join([f"{t:.2f}" for t in onsets_in_slot]) if has_onset else "None"

            prompt = (
                f"You are generating a StepMania beatmap (Beginner difficulty).\n"
                f"Measure: {M+1} / {total_measures} | Beat: {b+1}/4\n"
                f"Global BPM: {global_bpm:.1f}\n"
                f"Current Beat Time: {beat_time:.2f}s\n\n"
            )

            if has_onset:
                prompt += "CONDITION: ONSET DETECTED. Select a valid 4-character step (e.g. 0001, 0110). Do NOT output 0000.\n"
            else:
                prompt += "CONDITION: NO ONSET here. You MUST output 0000.\n"

            prompt += "\nHISTORY (All Previous Measures):\n"
            prompt += "\n".join(history_log) if history_log else "None."
            prompt += "\n\n---> Please output ONLY the 4-character string for THIS beat."

            beat_start_time_calc = time.time()

            if not has_onset:
                selected_step = "0000"
                probs_dict = {c: (100.0 if c == "0000" else 0.0) for c in ALL_16_COMBOS}
                calc_time = time.time() - beat_start_time_calc
            else:
                recent_history = global_linear_history[-8:]

                probs_dict = get_deepresonance_16_step_probabilities(
                    tmp_path, prompt, ALL_16_COMBOS,
                    temperature=1.8,
                    top_p=0.9,
                    min_p=0.01,
                    top_k=None,
                    repetition_penalty=1.5,
                    recent_history=recent_history
                )

                calc_time = time.time() - beat_start_time_calc

                choices = list(probs_dict.keys())
                weights = list(probs_dict.values())
                selected_step = random.choices(choices, weights=weights)[0]

            prob_format = "|".join([f"{k}-{v:.1f}%" for k, v in probs_dict.items()])

            csv_rows.append([
                f"{global_bpm:.1f}",
                f"{duration_min:.2f}",
                f"{beat_time:.2f}",
                onsets_timestamps_str,
                has_onset,
                selected_step,
                prob_format,
                selected_step,
                f"{calc_time:.2f}"
            ])

            txt_rows.append(selected_step)
            measure_step_outputs.append(selected_step)
            global_linear_history.append(selected_step)

        history_log.append(f"Measure {M+1}: " + ", ".join(measure_step_outputs))

        os.remove(tmp_path)
        gc.collect(); torch.cuda.empty_cache()

    txt_rows.append(",")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"DeepResonance_MeasureGrid_Wonsets_Wtempo_WBPM_{ts}.csv")
    txt_path = os.path.join(OUT_DIR, f"DeepResonance_MeasureGrid_Wonsets_Wtempo_WBPM_{ts}.txt")

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_rows))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# ML Generation Strategy", "Weighted Multinomial Probability Sampling"])
        w.writerow(["# Graph Search Strategy", "Exhaustive Sequence Evaluation via DeepResonance predict()"])
        w.writerow(["# Temperature", "1.8"])
        w.writerow(["# Top-P", "0.9"])
        w.writerow(["# Min-P", "0.01"])
        w.writerow(["# Top-K", "None"])
        w.writerow(["# Repetition Penalty", "Dynamic Scale Matrix (1.0 + N*0.5, Last-Beat Multiplier 1.1x)"])
        w.writerow([
            "global_bpm", "song_duration_minutes", "beat_time_sec",
            "onset_timestamps_in_beat", "onset_detected", "selected_step",
            "step_probabilities", "final_step_selection", "time_taken_sec"
        ])
        for r in csv_rows:
            w.writerow(r)

    print(f"✅ Generated {len(csv_rows)} rows across {total_measures} measures.")
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
