#!/usr/bin/env python3
"""
CSV Beatmap Validator

Validates a Gemini-generated CSV beatmap against:
  1. Musical features (onsets, beats, percussive) from the audio
  2. The original .ssc Hard chart (side-by-side comparison)

Usage:
  Edit the CONFIG section below, then run:
    .venv1/bin/python validate_csv_beatmap.py

Output:
  - Console report
  - validation_csv_<timestamp>.html  — interactive comparison chart
"""

import sys
import os
import re
import csv
import glob
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.beatmap_validator import (
    load_beatmap_measures,
    get_step_times,
    detect_onsets,
    detect_beats,
    extract_percussive_onsets,
    calculate_alignment,
    ValidationResults,
    print_validation_report,
    print_comparison_report,
    save_results_json,
)

# ============================================================================
# USER CONFIGURATION (EDIT THIS SECTION)
# ============================================================================

# 1. Generated CSV beatmap produced by batch_process_fraxtil.py
CSV_BEATMAP_PATH = (
    "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/"
    "Fraxtil's Arrow Arrangements/Love to Rise in the Summer Morning/"
    "Love to Rise in the Summer Morning_Hard_gemini-pro-latest_20260219_053846.csv"
)

# 2. Original .ssc file for the same song (used for comparison + BPM/offset)
ORIGINAL_SSC_PATH = (
    "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/"
    "Fraxtil's Arrow Arrangements/Love to Rise in the Summer Morning/"
    "Love to Rise in the Summer Morning.ssc"
)

# 3. Audio file — set to None to auto-detect from the same folder as the CSV
AUDIO_FILE_PATH = None

# 4. Which difficulty to pull from the .ssc for comparison
DIFFICULTY = "Hard"

# 5. Tolerance window for "on-beat" alignment
TOLERANCE_MS = 50.0

# ============================================================================
# END CONFIGURATION
# ============================================================================

AUDIO_EXTENSIONS = ['.ogg', '.mp3', '.wav', '.flac', '.m4a', '.opus', '.mp4']


# ─── Helpers ─────────────────────────────────────────────────────────────────

def find_audio(folder: str) -> Optional[str]:
    for ext in AUDIO_EXTENSIONS:
        matches = glob.glob(os.path.join(folder, f"*{ext}"))
        if matches:
            return matches[0]
    return None


def read_ssc_metadata(ssc_path: str) -> Tuple[float, float, Optional[str]]:
    """Return (bpm, offset, audio_filename) from a .ssc file."""
    bpm, offset, audio_filename = 120.0, 0.0, None
    with open(ssc_path, 'r', errors='ignore') as f:
        content = f.read()
    m = re.search(r'#BPMS:.*?=([\d.]+)', content, re.DOTALL)
    if m:
        bpm = float(m.group(1))
    m = re.search(r'#OFFSET:(-?[\d.]+)', content)
    if m:
        offset = float(m.group(1))
    m = re.search(r'#MUSIC:(.*?);', content)
    if m:
        audio_filename = m.group(1).strip()
    return bpm, offset, audio_filename


# ─── CSV Loading ─────────────────────────────────────────────────────────────

def load_csv_beatmap(csv_path: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load the generated CSV and return step timestamps + full row metadata.
    Skips separator rows (notes==',') and empty rows (notes=='0000').
    """
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes = row.get('notes', '').strip().strip('"')
            if notes in (',', '0000', ''):
                continue
            try:
                rows.append({
                    'time_ms': float(row['time_ms']),
                    'beat_position': float(row.get('beat_position', 0)),
                    'notes': notes,
                    'placement_type': int(row.get('placement_type', 0)),
                    'note_type': int(row.get('note_type', 2)),
                    'confidence': float(row.get('confidence', 1.0)),
                    'instrument': row.get('instrument', 'unknown').strip(),
                })
            except (ValueError, KeyError):
                continue

    times = np.array([r['time_ms'] / 1000.0 for r in rows])  # convert to seconds
    return times, rows


# ─── Validation ──────────────────────────────────────────────────────────────

def _classify_steps(
    step_times: np.ndarray,
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    perc_times: np.ndarray,
    audio_duration: float,
    tolerance_ms: float,
) -> Dict:
    """Classify each step and return counts + per-step distances."""
    t = tolerance_ms / 1000.0

    _, _, d_onset = calculate_alignment(step_times, onset_times, tolerance_ms)
    _, _, d_beat  = calculate_alignment(step_times, beat_times,  tolerance_ms)
    _, _, d_perc  = calculate_alignment(step_times, perc_times,  tolerance_ms)

    d_onset = np.array(d_onset) if d_onset else np.full(len(step_times), 9.9)
    d_beat  = np.array(d_beat)  if d_beat  else np.full(len(step_times), 9.9)
    d_perc  = np.array(d_perc)  if d_perc  else np.full(len(step_times), 9.9)

    perfect = beat = onset = perc = bad = oob = 0
    for i, st in enumerate(step_times):
        if audio_duration > 0 and st > audio_duration:
            oob += 1
            continue
        ib = d_beat[i]  <= t
        io = d_onset[i] <= t
        ip = d_perc[i]  <= t
        if ib and io:   perfect += 1
        elif ib:        beat    += 1
        elif io:        onset   += 1
        elif ip:        perc    += 1
        else:           bad     += 1

    valid = perfect + beat + onset + perc + bad
    return dict(
        perfect=perfect, beat=beat, onset=onset, perc=perc, bad=bad, oob=oob,
        valid=valid,
        d_onset=d_onset, d_beat=d_beat, d_perc=d_perc,
    )


def validate_from_csv(
    csv_path: str,
    audio_path: str,
    tolerance_ms: float = 50.0,
) -> ValidationResults:
    """Validate the generated CSV beatmap against audio features."""
    print(f"\n{'='*60}")
    print("VALIDATING GENERATED CSV BEATMAP")
    print(f"{'='*60}")
    print(f"CSV:   {os.path.basename(csv_path)}")
    print(f"Audio: {os.path.basename(audio_path)}")

    print("\n[1/4] Loading CSV…")
    step_times, csv_rows = load_csv_beatmap(csv_path)
    total_arrows = sum(row['notes'].count('1') for row in csv_rows)
    print(f"  {len(step_times)} active steps, {total_arrows} arrows")

    print("\n[2/4] Detecting onsets…")
    onset_times, sr = detect_onsets(audio_path)
    audio_duration = float(sr) and __import__('librosa').get_duration(path=audio_path)
    print(f"  {len(onset_times)} onsets  |  duration {audio_duration:.1f}s")

    print("\n[3/4] Detecting beats…")
    beat_times, tempo = detect_beats(audio_path)
    print(f"  {len(beat_times)} beats  |  tempo {tempo:.1f} BPM")

    print("\n[4/4] Extracting percussive onsets (fast mode)…")
    import librosa as _librosa
    _y, _sr = _librosa.load(audio_path, sr=22050)
    _onset_env = _librosa.onset.onset_strength(y=_y, sr=_sr, aggregate=np.median)
    _onset_frames = _librosa.onset.onset_detect(
        onset_envelope=_onset_env, sr=_sr, backtrack=True,
        pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.07, wait=5
    )
    perc_times = _librosa.frames_to_time(_onset_frames, sr=_sr)
    print(f"  {len(perc_times)} percussive onsets")



    c = _classify_steps(step_times, onset_times, beat_times, perc_times,
                        audio_duration, tolerance_ms)
    v = c['valid']

    onset_pct = ((c['perfect'] + c['onset']) / v * 100) if v else 0.0
    beat_pct  = ((c['perfect'] + c['beat'])  / v * 100) if v else 0.0
    perc_pct  = ((c['perfect'] + c['perc'])  / v * 100) if v else 0.0

    # Build partials from beat_position
    partials = []
    for row in csv_rows:
        frac = row['beat_position'] % 1.0
        eps = 0.01
        if frac < eps or frac > 1 - eps:          partials.append("4th Note (Red)")
        elif abs(frac - 0.5) < eps:                partials.append("8th Note (Blue)")
        elif abs(frac - 0.25) < eps or abs(frac - 0.75) < eps: partials.append("16th Note (Yellow)")
        elif abs(frac - 1/3) < eps or abs(frac - 2/3) < eps:   partials.append("12th Note (Purple)")
        else:                                      partials.append("Other")

    return ValidationResults(
        onset_alignment_percentage=onset_pct,
        steps_on_onsets=c['perfect'] + c['onset'],
        total_steps=len(step_times),
        mean_onset_distance_ms=float(np.mean(c['d_onset']) * 1000) if len(c['d_onset']) else 0,
        beat_alignment_percentage=beat_pct,
        steps_on_beats=c['perfect'] + c['beat'],
        mean_beat_distance_ms=float(np.mean(c['d_beat']) * 1000) if len(c['d_beat']) else 0,
        percussive_alignment_percentage=perc_pct,
        steps_on_percussive=c['perfect'] + c['perc'],
        onset_times=onset_times.tolist(),
        step_times=step_times.tolist(),
        beat_times=beat_times.tolist(),
        distances_to_onsets=c['d_onset'].tolist(),
        distances_to_beats=c['d_beat'].tolist(),
        distances_to_percussive=c['d_perc'].tolist(),
        step_row_indices=[f"row{i}" for i in range(len(step_times))],
        step_row_contents=[r['notes'] for r in csv_rows],
        step_partials=partials,
        audio_file=audio_path,
        beatmap_file=csv_path,
        bpm=tempo,
        offset=0.0,
        timestamp=datetime.now().isoformat(),
        total_arrows=total_arrows,
        match_perfect=c['perfect'],
        match_beat=c['beat'],
        match_onset=c['onset'],
        match_perc=c['perc'],
        match_bad=c['bad'],
        match_out_of_bounds=c['oob'],
        audio_duration=audio_duration,
    )


# ─── Visualization ───────────────────────────────────────────────────────────

# Colour palette (matches the legend image)
COLOURS = {
    'perfect':  '#00BFFF',   # cyan
    'beat':     '#0000FF',   # blue
    'onset':    '#800080',   # purple
    'perc':     '#FFA500',   # orange
    'bad':      '#FF0000',   # red  (X marker)
    'oob':      '#808080',   # grey (crossed circle)
    'onset_line': '#888888', # grey vertical line
    'beat_line':  '#0000FF', # blue vertical line
}


def _step_color(results: ValidationResults):
    """Return per-step colour + symbol arrays based on classification."""
    t = 50.0 / 1000.0  # 50ms in seconds
    d_onset = np.array(results.distances_to_onsets) if results.distances_to_onsets else np.full(results.total_steps, 9.9)
    d_beat  = np.array(results.distances_to_beats)  if results.distances_to_beats  else np.full(results.total_steps, 9.9)
    d_perc  = np.array(results.distances_to_percussive) if results.distances_to_percussive else np.full(results.total_steps, 9.9)

    colors, symbols = [], []
    step_times = np.array(results.step_times)

    for i in range(len(step_times)):
        if results.audio_duration > 0 and step_times[i] > results.audio_duration:
            colors.append(COLOURS['oob']); symbols.append('x-open'); continue
        ib = d_beat[i]  <= t
        io = d_onset[i] <= t
        ip = d_perc[i]  <= t
        if ib and io:   colors.append(COLOURS['perfect']); symbols.append('circle')
        elif ib:        colors.append(COLOURS['beat']);    symbols.append('circle')
        elif io:        colors.append(COLOURS['onset']);   symbols.append('circle')
        elif ip:        colors.append(COLOURS['perc']);    symbols.append('circle')
        else:           colors.append(COLOURS['bad']);     symbols.append('x')
    return colors, symbols


def _add_panel(fig, results: ValidationResults, col: int, title: str):
    """Add onset/beat/step traces to one subplot column."""
    step_times = np.array(results.step_times)
    colors, symbols = _step_color(results)

    # Onsets — thin grey vertical lines
    for ot in results.onset_times:
        fig.add_shape(type='line', x0=ot, x1=ot, y0=3.6, y1=4.4,
                      line=dict(color=COLOURS['onset_line'], width=1),
                      row=1, col=col)

    # Beats — blue vertical lines (taller)
    for bt in results.beat_times:
        fig.add_shape(type='line', x0=bt, x1=bt, y0=2.6, y1=3.4,
                      line=dict(color=COLOURS['beat_line'], width=2),
                      row=1, col=col)

    # Step dots — coloured by classification
    hover = [
        f"t={st:.3f}s<br>{results.step_row_contents[i] if results.step_row_contents else ''}<br>{results.step_partials[i] if results.step_partials else ''}"
        for i, st in enumerate(step_times)
    ]

    fig.add_trace(go.Scatter(
        x=step_times, y=[2.0] * len(step_times),
        mode='markers',
        marker=dict(color=colors, symbol=symbols, size=10,
                    line=dict(color='white', width=1)),
        name=f'{title} Steps',
        hovertext=hover, hoverinfo='text',
    ), row=1, col=col)

    # Dummy traces for legend (first panel only)
    if col == 1:
        legend_items = [
            ('Onsets',           COLOURS['onset_line'], 'line-ns-open'),
            ('Beats',            COLOURS['beat_line'],  'line-ns-open'),
            ('Perfect (Beat+Onset)', COLOURS['perfect'], 'circle'),
            ('Grid (Beat)',      COLOURS['beat'],       'circle'),
            ('Audio (Onset)',    COLOURS['onset'],      'circle'),
            ('Percussive',       COLOURS['perc'],       'circle'),
            ('Unaligned (Bad)',  COLOURS['bad'],        'x'),
            ('Out of Bounds',    COLOURS['oob'],        'x-open'),
        ]
        for label, color, sym in legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color=color, symbol=sym, size=10),
                name=label, showlegend=True,
            ), row=1, col=col)


def _summary_table(results: ValidationResults, label: str) -> str:
    v = results.total_steps - results.match_out_of_bounds
    rows = [
        ("Steps (valid)", f"{v}"),
        ("Total Arrows",  f"{results.total_arrows}"),
        ("Perfect (Beat+Onset)", f"{results.match_perfect}"),
        ("Grid (Beat)",   f"{results.match_beat}"),
        ("Audio (Onset)", f"{results.match_onset}"),
        ("Percussive",    f"{results.match_perc}"),
        ("Unaligned",     f"{results.match_bad}"),
        ("Out of Bounds", f"{results.match_out_of_bounds}"),
        ("Onset Align %", f"{results.onset_alignment_percentage:.1f}%"),
        ("Beat Align %",  f"{results.beat_alignment_percentage:.1f}%"),
        ("Percussive %",  f"{results.percussive_alignment_percentage:.1f}%"),
    ]
    lines = [f"<b>{label}</b>"]
    for k, v in rows:
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def visualize_comparison(
    original: ValidationResults,
    generated: ValidationResults,
    output_path: str,
):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original (.ssc)", "Generated (CSV)"],
        horizontal_spacing=0.06,
    )

    _add_panel(fig, original,  col=1, title="Original")
    _add_panel(fig, generated, col=2, title="Generated")

    # Summary annotations
    orig_text = _summary_table(original,  "Original")
    gen_text  = _summary_table(generated, "Generated")

    fig.add_annotation(text=orig_text, xref="paper", yref="paper",
                       x=0.02, y=-0.18, showarrow=False, align="left",
                       font=dict(size=11), bordercolor="#ccc", borderwidth=1,
                       bgcolor="white")
    fig.add_annotation(text=gen_text,  xref="paper", yref="paper",
                       x=0.55, y=-0.18, showarrow=False, align="left",
                       font=dict(size=11), bordercolor="#ccc", borderwidth=1,
                       bgcolor="white")

    audio_dur = max(original.audio_duration, generated.audio_duration)
    fig.update_layout(
        title=dict(text="Beatmap Validation — Original vs Generated", font=dict(size=16)),
        height=500,
        xaxis=dict(title="Time (s)", range=[0, audio_dur]),
        xaxis2=dict(title="Time (s)", range=[0, audio_dur]),
        yaxis=dict(showticklabels=False, range=[1, 5]),
        yaxis2=dict(showticklabels=False, range=[1, 5]),
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(b=220),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    fig.write_html(output_path)
    print(f"\nVisualization saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(" CSV BEATMAP VALIDATOR")
    print("="*60)

    # ── 1. Check CSV file ──────────────────────────────────────────────────
    if not os.path.exists(CSV_BEATMAP_PATH):
        print(f"\nError: CSV file not found:\n  {CSV_BEATMAP_PATH}")
        return

    # ── 2. Resolve audio ──────────────────────────────────────────────────
    audio_path = AUDIO_FILE_PATH
    if not audio_path or not os.path.exists(audio_path):
        # Try same folder as CSV first, then SSC folder
        for folder in [os.path.dirname(CSV_BEATMAP_PATH),
                       os.path.dirname(ORIGINAL_SSC_PATH) if ORIGINAL_SSC_PATH else None]:
            if folder:
                found = find_audio(folder)
                if found:
                    audio_path = found
                    print(f"Auto-detected audio: {os.path.basename(audio_path)}")
                    break

    if not audio_path or not os.path.exists(audio_path):
        print("Error: could not find audio file. Set AUDIO_FILE_PATH explicitly.")
        return

    # ── 3. Read BPM/offset from SSC (for original chart validation) ───────
    bpm, offset = 120.0, 0.0
    if ORIGINAL_SSC_PATH and os.path.exists(ORIGINAL_SSC_PATH):
        bpm, offset, _ = read_ssc_metadata(ORIGINAL_SSC_PATH)
        print(f"SSC metadata → BPM={bpm}, Offset={offset}s")

    # ── 4. Validate Generated CSV ─────────────────────────────────────────
    gen_results = validate_from_csv(CSV_BEATMAP_PATH, audio_path, TOLERANCE_MS)
    print_validation_report(gen_results)

    # ── 5. Validate Original SSC ──────────────────────────────────────────
    orig_results = None
    if ORIGINAL_SSC_PATH and os.path.exists(ORIGINAL_SSC_PATH):
        print(f"\n{'='*60}")
        print(f"VALIDATING ORIGINAL SSC ({DIFFICULTY})")
        print(f"{'='*60}")
        from analysis.beatmap_validator import validate_beatmap
        orig_results = validate_beatmap(
            audio_path=audio_path,
            beatmap_path=ORIGINAL_SSC_PATH,
            bpm=bpm,
            offset=offset,
            tolerance_ms=TOLERANCE_MS,
            difficulty=DIFFICULTY,
        )
        print_validation_report(orig_results)
        print_comparison_report(orig_results, gen_results)

    # ── 6. Save outputs ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(CSV_BEATMAP_PATH)

    json_path = os.path.join(out_dir, f"validation_csv_{ts}.json")
    save_results_json(gen_results, json_path)

    html_path = os.path.join(out_dir, f"validation_csv_{ts}.html")
    if orig_results:
        visualize_comparison(orig_results, gen_results, html_path)
    else:
        from analysis.beatmap_validator import visualize_validation
        visualize_validation(gen_results, html_path)

    print(f"\n✅ Done!  Open in browser: {html_path}")


if __name__ == "__main__":
    main()
