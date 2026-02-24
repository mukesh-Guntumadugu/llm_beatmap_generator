"""
Beatmap Validation System

This module provides comprehensive validation of beatmap step placements against
musical features including onsets, beats, and percussive elements.

Features:
1. Onset alignment validation - check if steps are placed on musical onsets
2. Beat alignment validation - check if steps align with detected beats
3. Feature-based placement - validate steps against drums/percussive elements
4. Comparison mode - compare original vs generated beatmaps
5. Detailed reporting - JSON, text, and visualizations
"""

import numpy as np
import librosa
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
from datetime import datetime


@dataclass
class ValidationResults:
    """Container for validation results"""
    # Onset alignment
    onset_alignment_percentage: float
    steps_on_onsets: int
    total_steps: int
    mean_onset_distance_ms: float
    
    # Beat alignment
    beat_alignment_percentage: float
    steps_on_beats: int
    mean_beat_distance_ms: float
    
    # Feature-based placement
    percussive_alignment_percentage: float
    steps_on_percussive: int
    
    # Additional metrics
    onset_times: List[float]
    step_times: List[float]
    beat_times: List[float]
    distances_to_onsets: List[float]
    distances_to_beats: List[float]
    distances_to_percussive: List[float]
    
    # Metadata
    audio_file: str
    beatmap_file: str
    bpm: float
    offset: float
    timestamp: str
    total_arrows: int = 0
    
    # Detailed step metadata for interactive visualization
    step_row_indices: List[str] = None
    step_row_contents: List[str] = None
    step_partials: List[str] = None
    
    # Detailed classification counts
    match_perfect: int = 0   # Beat + Onset
    match_beat: int = 0      # Beat only
    match_onset: int = 0     # Onset only
    match_perc: int = 0      # Percussive only
    match_bad: int = 0       # Unaligned
    match_out_of_bounds: int = 0 # Beyond Audio End

    
    # Audio Duration
    audio_duration: float = 0.0


def load_beatmap_from_ssc(filepath: str, difficulty: str = "Easy") -> List[List[str]]:
    """
    Load beatmap measures from an .ssc file for a specific difficulty.
    """
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()

    # Split by charts (charts start with #NOTEDATA:;)
    charts = content.split('#NOTEDATA:;')
    
    target_chart = None
    
    for chart in charts:
        if f"#DIFFICULTY:{difficulty};" in chart or f"#DIFFICULTY:{difficulty}\n" in chart:
            target_chart = chart
            break
            
    if not target_chart:
        # Fallback: try to find any chart if specific one not found
        # Or look for "Challenge" if "Easy" missing (since we saw Challenge in the preview)
        print(f"⚠️ Warning: Difficulty '{difficulty}' not found. Looking for 'Challenge' or first available...")
        for chart in charts:
            if "#NOTES:" in chart:
                target_chart = chart
                break
    
    if not target_chart:
        print("❌ Error: No chart data found in .ssc file")
        return []

    # Extract notes section
    # #NOTES:\n measure data... ;
    try:
        notes_part = target_chart.split('#NOTES:')[1]
        # Notes end with a semicolon
        notes_data = notes_part.split(';')[0].strip()
    except IndexError:
        print("Error: Malformed #NOTES section")
        return []

    measures = []
    current_measure = []
    
    for line in notes_data.split('\n'):
        line = line.strip().split('//')[0].strip() # Remove comments
        if not line:
            continue
            
        if line == ',':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        elif len(line) == 4 and all(c in '01234M' for c in line):
            current_measure.append(line)
            
    if current_measure:
        measures.append(current_measure)
        
    return measures


def load_beatmap_measures(filepath: str, difficulty: str = "Easy") -> List[List[str]]:
    """
    Load beatmap measures from file.
    
    Args:
        filepath: Path to beatmap file (.txt, .text, or .ssc)
        difficulty: Difficulty to extract if using .ssc (default "Easy")
        
    Returns:
        List of measures, where each measure is a list of note rows
    """
    if filepath.lower().endswith('.ssc'):
        return load_beatmap_from_ssc(filepath, difficulty)

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if not content.strip():
                print(f"Warning: File {os.path.basename(filepath)} is empty!")
                return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    measures = []
    current_measure = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line == ',':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        elif len(line) == 4 and all(c in '01234M' for c in line):
            current_measure.append(line)
            
    if current_measure:
        measures.append(current_measure)
        
    return measures


def get_step_times(measures: List[List[str]], bpm: float, offset: float) -> Tuple[np.ndarray, int, List[str], List[str], List[str]]:
    """
    Convert beatmap measures to timestamps and count arrows.
    
    Args:
        measures: List of beatmap measures
        bpm: Beats per minute
        offset: Time offset in seconds where first beat occurs
        
    Returns:
        Tuple of (step_times, total_arrows, step_indices, step_contents, step_partials)
    """
    step_times = []
    step_indices = []
    step_contents = []
    step_partials = []
    total_arrows = 0
    beats_per_measure = 4  # Assuming 4/4 time signature
    seconds_per_beat = 60.0 / bpm
    
    start_time = offset
    
    for measure_idx, measure in enumerate(measures):
        measure_start_beat = measure_idx * beats_per_measure
        lines_in_measure = len(measure)
        if lines_in_measure == 0:
            continue
        
        beats_per_line = beats_per_measure / lines_in_measure
        
        for line_idx, line in enumerate(measure):
            # Check if there is a note (any non-zero, excluding mines for now)
            # Count arrows: 1 (Tap), 2 (Hold Head), 4 (Roll Head)
            arrow_count = sum(1 for c in line if c in '124')
            
            # Only consider rows with note HEADS as steps to be validated
            # Ignore '3' (Hold Tail) and 'M' (Mine) for timing validation
            has_note = any(c in '124' for c in line)
            
            if has_note:
                beat_time = measure_start_beat + (line_idx * beats_per_line)
                timestamp = start_time + (beat_time * seconds_per_beat)
                step_times.append(timestamp)
                total_arrows += arrow_count
                
                # Capture metadata
                step_contents.append(line)
                step_indices.append(f"M{measure_idx}:L{line_idx}") # Measure:Line format
                
                # Determine Partial (Snap)
                frac = beat_time % 1.0
                epsilon = 0.001
                partial = "Other"
                
                if abs(frac) < epsilon or abs(frac - 1.0) < epsilon:
                    partial = "4th Note (Red)"
                elif abs(frac - 0.5) < epsilon:
                    partial = "8th Note (Blue)"
                elif abs(frac - 0.25) < epsilon or abs(frac - 0.75) < epsilon:
                    partial = "16th Note (Yellow)"
                elif abs(frac - 1/3) < epsilon or abs(frac - 2/3) < epsilon:
                    partial = "12th Note (Purple)"
                elif abs(frac - 1/6) < epsilon or abs(frac - 5/6) < epsilon:
                    partial = "24th Note (Pink)"
                elif abs(frac - 1/8) < epsilon or abs(frac - 3/8) < epsilon or abs(frac - 5/8) < epsilon or abs(frac - 7/8) < epsilon:
                    partial = "32nd Note (Orange)"
                elif abs(frac - 1/48) < epsilon: # Roughly check for 48ths
                    partial = "48th Note (Cyan)"
                else:
                     partial = "Unquantized"
                
                step_partials.append(partial)
                
    return np.array(step_times), total_arrows, step_indices, step_contents, step_partials


def detect_onsets(audio_path: str, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Detect musical onsets in audio.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Additional arguments for librosa.onset.onset_detect
        
    Returns:
        Tuple of (onset_times, sample_rate)
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Detect onsets with backtracking for better accuracy
    onset_frames = librosa.onset.onset_detect(
        y=y, 
        sr=sr, 
        backtrack=True,
        **kwargs
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    return onset_times, sr


def detect_beats(audio_path: str, start_bpm: float = None) -> Tuple[np.ndarray, float]:
    """
    Detect beats in audio.
    
    Args:
        audio_path: Path to audio file
        start_bpm: Optional hint for expected BPM
        
    Returns:
        Tuple of (beat_times, estimated_tempo)
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Detect beats
    # Use start_bpm if provided to help the beat tracker
    kwargs = {'start_bpm': start_bpm} if start_bpm else {}
    
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, **kwargs)
    except Exception:
        # Fallback without hint if it fails
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Handle tempo - it can be an array or scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    
    return beat_times, tempo


def extract_percussive_onsets(audio_path: str) -> np.ndarray:
    """
    Extract onsets from percussive component of audio.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Array of percussive onset times
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Detect onsets on percussive component
    onset_frames = librosa.onset.onset_detect(
        y=y_percussive,
        sr=sr,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    return onset_times



def calculate_alignment(
    step_times: np.ndarray,
    reference_times: np.ndarray,
    tolerance_ms: float = 50.0
) -> Tuple[float, int, List[float]]:
    """
    Calculate alignment percentage between steps and reference times.
    
    Args:
        step_times: Array of step timestamps
        reference_times: Array of reference timestamps (onsets/beats)
        tolerance_ms: Tolerance window in milliseconds
        
    Returns:
        Tuple of (alignment_percentage, aligned_count, distances_list)
    """
    if len(step_times) == 0 or len(reference_times) == 0:
        return 0.0, 0, []
    
    tolerance_sec = tolerance_ms / 1000.0
    aligned_count = 0
    distances = []
    
    for step_time in step_times:
        # Find nearest reference time
        idx = np.searchsorted(reference_times, step_time)
        
        min_distance = float('inf')
        
        # Check left neighbor
        if idx > 0:
            left_dist = abs(step_time - reference_times[idx - 1])
            min_distance = min(min_distance, left_dist)
        
        # Check right neighbor
        if idx < len(reference_times):
            right_dist = abs(step_time - reference_times[idx])
            min_distance = min(min_distance, right_dist)
        
        distances.append(min_distance)
        
        # Check if within tolerance
        if min_distance <= tolerance_sec:
            aligned_count += 1
    
    alignment_percentage = (aligned_count / len(step_times)) * 100.0
    
    return alignment_percentage, aligned_count, distances


def validate_beatmap(
    audio_path: str,
    beatmap_path: str,
    bpm: float,
    offset: float,
    tolerance_ms: float = 50.0,
    difficulty: str = "Easy"
) -> ValidationResults:
    """
    Validate beatmap step placements against musical features.
    
    Args:
        audio_path: Path to audio file
        beatmap_path: Path to beatmap file
        bpm: Beats per minute
        offset: Time offset in seconds
        tolerance_ms: Tolerance window for alignment (default 50ms)
        difficulty: Difficulty to extract if using .ssc (default "Easy")
        
    Returns:
        ValidationResults object with all metrics
    """
    print(f"\n{'='*60}")
    print(f"BEATMAP VALIDATION")
    print(f"{'='*60}")
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Beatmap: {os.path.basename(beatmap_path)}")
    print(f"BPM: {bpm}, Offset: {offset}s")
    print(f"Tolerance: {tolerance_ms}ms")
    
    # Load beatmap and get step times
    print("\n[1/5] Loading beatmap...")
    measures = load_beatmap_measures(beatmap_path, difficulty)
    step_times, total_arrows, step_indices, step_contents, step_partials = get_step_times(measures, bpm, offset)
    print(f"  Found {len(step_times)} rows with steps")
    print(f"  Total Arrows (Notes): {total_arrows}")
    
    # Detect onsets
    print("\n[2/5] Detecting onsets...")
    onset_times, sr = detect_onsets(audio_path)
    print(f"  Detected {len(onset_times)} onsets")
    
    # Get Audio Duration
    audio_duration = librosa.get_duration(path=audio_path)
    print(f"  Audio Duration: {audio_duration:.2f}s")
    
    # Detect beats
    print("\n[3/5] Detecting beats...")
    # Use known BPM as a hint to improve detection accuracy
    beat_times, estimated_tempo = detect_beats(audio_path, start_bpm=bpm)
    print(f"  Detected {len(beat_times)} beats (tempo: {estimated_tempo:.1f} BPM)")
    
    # Extract percussive onsets
    print("\n[4/5] Extracting percussive features...")
    percussive_onset_times = extract_percussive_onsets(audio_path)
    print(f"  Found {len(percussive_onset_times)} percussive onsets")
    
    # Calculate alignments
    print("\n[5/5] Calculating alignments...")
    
    # Onset alignment
    onset_align_pct, steps_on_onsets, onset_distances = calculate_alignment(
        step_times, onset_times, tolerance_ms
    )
    mean_onset_dist_ms = np.mean(onset_distances) * 1000.0 if onset_distances else 0.0
    
    # Beat alignment
    beat_align_pct, steps_on_beats, beat_distances = calculate_alignment(
        step_times, beat_times, tolerance_ms
    )
    mean_beat_dist_ms = np.mean(beat_distances) * 1000.0 if beat_distances else 0.0
    
    # Percussive alignment
    perc_align_pct, steps_on_percussive, perc_distances = calculate_alignment(
        step_times, percussive_onset_times, tolerance_ms
    )
    
    # Calculate detailed classification counts
    # Convert distances to arrays (seconds)
    step_times_arr = np.array(step_times)
    d_onset = np.array(onset_distances) if onset_distances else np.full(len(step_times), 999.0)
    d_beat = np.array(beat_distances) if beat_distances else np.full(len(step_times), 999.0)
    d_perc = np.array(perc_distances) if perc_distances else np.full(len(step_times), 999.0)
    
    threshold_sec = tolerance_ms / 1000.0
    
    match_perfect = 0
    match_beat = 0
    match_onset = 0
    match_perc = 0
    match_bad = 0
    match_out_of_bounds = 0
    
    # Track which steps are considered "valid" (within audio duration)
    valid_step_count = 0
    
    for i in range(len(step_times)):
        # Check if out of bounds
        if audio_duration > 0 and step_times[i] > audio_duration:
            match_out_of_bounds += 1
            # We treat this as a separate category, not "bad" in the alignment sense,
            # but "invalid" content.
            continue
            
        valid_step_count += 1
        
        is_beat = d_beat[i] <= threshold_sec
        is_onset = d_onset[i] <= threshold_sec
        is_perc = d_perc[i] <= threshold_sec
        
        if is_beat and is_onset:
            match_perfect += 1
        elif is_beat:
            match_beat += 1
        elif is_onset:
            match_onset += 1
        elif is_perc:
            match_perc += 1
        else:
            match_bad += 1

    # Re-calculate percentages based on VALID steps only
    # This prevents ghost steps at the end from skewing the "On Beat" score 
    # (either positively if they match grid, or negatively if they don't)
    if valid_step_count > 0:
        onset_align_pct = ((match_perfect + match_onset) / valid_step_count) * 100.0
        beat_align_pct = ((match_perfect + match_beat) / valid_step_count) * 100.0
        perc_align_pct = ((match_perfect + match_perc + match_onset) / valid_step_count) * 100.0 # Approximation
        # Actually perc_align_pct usually just counts steps matching percussive / total
        # Let's trust the logic: steps_on_percussive should be filtered too?
        # Yes, calculate_alignment return counts included ghosts. 
        # We should recalculate steps_on_X from match_X counts.
        steps_on_onsets = match_perfect + match_onset
        steps_on_beats = match_perfect + match_beat
        steps_on_percussive = match_perfect + match_perc # This is simpler, likely lower bound
        # Note: steps_on_percussive in calculate_alignment was just "close to percussive".
        # distinct buckets in counting loop make it mutually exclusive.
        # But 'steps_on_beats' usually means "is it on a beat (regardless of onset)?"
        # So steps_on_beats = match_perfect + match_beat is correct.
    else:
        onset_align_pct = 0.0
        beat_align_pct = 0.0
        perc_align_pct = 0.0

    # Create results object
    results = ValidationResults(
        onset_alignment_percentage=onset_align_pct,
        steps_on_onsets=steps_on_onsets,
        total_steps=len(step_times),
        mean_onset_distance_ms=mean_onset_dist_ms,
        
        beat_alignment_percentage=beat_align_pct,
        steps_on_beats=steps_on_beats,
        mean_beat_distance_ms=mean_beat_dist_ms,
        
        percussive_alignment_percentage=perc_align_pct,
        steps_on_percussive=steps_on_percussive,
        
        onset_times=onset_times.tolist(),
        step_times=step_times.tolist(),
        beat_times=beat_times.tolist(),
        distances_to_onsets=onset_distances,
        distances_to_beats=beat_distances,
        distances_to_percussive=perc_distances,
        
        # New Step Metadata
        step_row_indices=step_indices,
        step_row_contents=step_contents,
        step_partials=step_partials,
        
        audio_file=audio_path,
        beatmap_file=beatmap_path,
        bpm=bpm,
        offset=offset,
        timestamp=datetime.now().isoformat(),
        total_arrows=total_arrows,
        
        match_perfect=match_perfect,
        match_beat=match_beat,
        match_onset=match_onset,
        match_perc=match_perc,
        match_bad=match_bad,
        
        audio_duration=audio_duration,
        match_out_of_bounds=match_out_of_bounds
    )
    
    return results


def compare_beatmaps(
    audio_path: str,
    original_beatmap_path: str,
    generated_beatmap_path: str,
    bpm: float,
    offset: float,
    tolerance_ms: float = 50.0
) -> Tuple[ValidationResults, ValidationResults]:
    """
    Compare original and generated beatmaps.
    
    Args:
        audio_path: Path to audio file
        original_beatmap_path: Path to original beatmap
        generated_beatmap_path: Path to generated beatmap
        bpm: Beats per minute
        offset: Time offset in seconds
        tolerance_ms: Tolerance window for alignment
        
    Returns:
        Tuple of (original_results, generated_results)
    """
    print("\n" + "="*60)
    print("COMPARING BEATMAPS")
    print("="*60)
    
    print("\n>>> Validating ORIGINAL beatmap...")
    original_results = validate_beatmap(
        audio_path, original_beatmap_path, bpm, offset, tolerance_ms
    )
    
    print("\n>>> Validating GENERATED beatmap...")
    generated_results = validate_beatmap(
        audio_path, generated_beatmap_path, bpm, offset, tolerance_ms
    )
    
    return original_results, generated_results


def print_validation_report(results: ValidationResults):
    """
    Print a comprehensive validation report.
    
    Args:
        results: ValidationResults object
    """
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    
    valid_steps = results.total_steps - results.match_out_of_bounds
    
    print(f"\nOVERVIEW")
    print(f"  Total Steps (Rows): {results.total_steps}")
    print(f"  Valid Steps (In Audio): {valid_steps}")
    if results.match_out_of_bounds > 0:
        print(f"  Out of Bounds (Ignored): {results.match_out_of_bounds}")
        
    print(f"  Total Arrows (Notes): {results.total_arrows}")
    print(f"  BPM: {results.bpm}")
    print(f"  Offset: {results.offset}s")
    
    print(f"\nSTEP CLASSIFICATION (Valid Steps Only)")
    print(f"  Perfect Match (Beat+Onset): {results.match_perfect}")
    print(f"  Grid Match (Beat only): {results.match_beat}")
    print(f"  Audio Match (Onset only): {results.match_onset}")
    print(f"  Percussive Match (Perc only): {results.match_perc}")
    print(f"  Unaligned (Bad/Ghost): {results.match_bad}")
    
    print(f"\nONSET ALIGNMENT")
    print(f"  Percentage on Onsets: {results.onset_alignment_percentage:.2f}%")
    print(f"  Steps on Onsets: {results.steps_on_onsets}/{valid_steps}")
    print(f"  Mean Distance to Nearest Onset: {results.mean_onset_distance_ms:.2f}ms")
    
    # Rating
    if results.onset_alignment_percentage >= 80:
        rating = "EXCELLENT"
    elif results.onset_alignment_percentage >= 60:
        rating = "GOOD"
    elif results.onset_alignment_percentage >= 40:
        rating = "FAIR"
    else:
        rating = "POOR"
    print(f"  Rating: {rating}")
    
    print(f"\nBEAT ALIGNMENT")
    print(f"  Percentage on Beats: {results.beat_alignment_percentage:.2f}%")
    print(f"  Steps on Beats: {results.steps_on_beats}/{valid_steps}")
    print(f"  Mean Distance to Nearest Beat: {results.mean_beat_distance_ms:.2f}ms")
    
    # Rating
    if results.beat_alignment_percentage >= 80:
        rating = "EXCELLENT"
    elif results.beat_alignment_percentage >= 60:
        rating = "GOOD"
    elif results.beat_alignment_percentage >= 40:
        rating = "FAIR"
    else:
        rating = "POOR"
    print(f"  Rating: {rating}")
    
    print(f"\nPERCUSSIVE ALIGNMENT")
    print(f"  Percentage on Percussive Features: {results.percussive_alignment_percentage:.2f}%")
    print(f"  Steps on Percussive: {results.steps_on_percussive}/{results.total_steps}")
    
    # Quantization Summary
    if results.step_partials:
        print(f"\nQUANTIZATION ANALYSIS (Rhythm)")
        from collections import Counter
        counts = Counter(results.step_partials)
        total = len(results.step_partials)
        
        # Sort by count descending
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        dominant = sorted_counts[0][0] if sorted_counts else "None"
        print(f"  Dominant Rhythm: {dominant}")
        
        for partial, count in sorted_counts:
            pct = (count / total) * 100
            print(f"  - {partial}: {count} ({pct:.1f}%)")
            
        print(f"  Most steps are {dominant} notes.")
    
    # Rating
    if results.percussive_alignment_percentage >= 70:
        rating = "EXCELLENT"
    elif results.percussive_alignment_percentage >= 50:
        rating = "GOOD"
    elif results.percussive_alignment_percentage >= 30:
        rating = "FAIR"
    else:
        rating = "POOR"
    print(f"  Rating: {rating}")
    
    print(f"\n{'='*60}")


def print_comparison_report(
    original: ValidationResults,
    generated: ValidationResults
):
    """
    Print a comparison report between original and generated beatmaps.
    
    Args:
        original: Validation results for original beatmap
        generated: Validation results for generated beatmap
    """
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")
    
    print(f"\nOVERVIEW")
    print(f"  Original Steps: {original.total_steps}")
    print(f"  Generated Steps: {generated.total_steps}")
    print(f"  Difference: {generated.total_steps - original.total_steps:+d}")
    
    print(f"\nONSET ALIGNMENT COMPARISON")
    print(f"  Original:  {original.onset_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.onset_alignment_percentage:6.2f}%")
    diff = generated.onset_alignment_percentage - original.onset_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\nBEAT ALIGNMENT COMPARISON")
    print(f"  Original:  {original.beat_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.beat_alignment_percentage:6.2f}%")
    diff = generated.beat_alignment_percentage - original.beat_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\nPERCUSSIVE ALIGNMENT COMPARISON")
    print(f"  Original:  {original.percussive_alignment_percentage:6.2f}%")
    print(f"  Generated: {generated.percussive_alignment_percentage:6.2f}%")
    diff = generated.percussive_alignment_percentage - original.percussive_alignment_percentage
    print(f"  Difference: {diff:+6.2f}%")
    
    print(f"\n{'='*60}")


def save_results_json(results: ValidationResults, output_path: str):
    """
    Save validation results to JSON file.
    
    Args:
        results: ValidationResults object
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")


def visualize_validation(
    results: ValidationResults,
    output_path: str = "validation_visualization.html"
):
    """
    Create interactive visualization of validation results using Plotly.
    
    Args:
        results: ValidationResults object
        output_path: Path to save visualization (should be .html)
    """
    # Ensure extension is .html
    if output_path.endswith('.png'):
        output_path = output_path.replace('.png', '.html')
        
    fig = go.Figure()

    # 1. Onsets
    fig.add_trace(go.Scatter(
        x=results.onset_times,
        y=[4] * len(results.onset_times),
        mode='markers',
        marker=dict(symbol='line-ns-open', size=15, color='gray', line=dict(width=2)),
        name='Onsets',
        hoverinfo='x'
    ))

    # 2. Beats
    fig.add_trace(go.Scatter(
        x=results.beat_times,
        y=[3] * len(results.beat_times),
        mode='markers',
        marker=dict(symbol='line-ns-open', size=20, color='blue', line=dict(width=3)),
        name='Beats',
        hoverinfo='x'
    ))

    
    
    # 3. Steps Classification (Alignment)
    step_times = np.array(results.step_times)
    
    # Calculate categories again (or use stored if complex logic)
    d_onset = np.array(results.distances_to_onsets) if results.distances_to_onsets else np.full(len(step_times), 999.0)
    d_beat = np.array(results.distances_to_beats) if results.distances_to_beats else np.full(len(step_times), 999.0)
    d_perc = np.array(results.distances_to_percussive) if results.distances_to_percussive else np.full(len(step_times), 999.0)
    
    threshold_sec = 0.050 # 50ms
    
    # Categories
    indices_perfect = []
    indices_beat = []
    indices_onset = []
    indices_perc = []
    indices_bad = []
    indices_out_of_bounds = []
    
    for i in range(len(step_times)):
        # Check if out of bounds
        if results.audio_duration > 0 and step_times[i] > results.audio_duration:
            indices_out_of_bounds.append(i)
            continue
            
        is_beat = d_beat[i] <= threshold_sec
        is_onset = d_onset[i] <= threshold_sec
        is_perc = d_perc[i] <= threshold_sec
        
        if is_beat and is_onset:
            indices_perfect.append(i)
        elif is_beat:
            indices_beat.append(i)
        elif is_onset:
            indices_onset.append(i)
        elif is_perc:
            indices_perc.append(i)
        else:
            indices_bad.append(i)
            
    # Helper to add trace
    def add_step_trace(indices, name, color, y_val, symbol='circle'):
        if not indices:
            return
        
        times = step_times[indices]
        
        # Metadata for hover
        custom_data = []
        if results.step_row_indices and results.step_row_contents:
            for idx in indices:
                row_idx = results.step_row_indices[idx]
                content = results.step_row_contents[idx]
                partial = results.step_partials[idx] if results.step_partials else "Unknown"
                custom_data.append([row_idx, content, partial])
        else:
            # Fallback if metadata missing
            custom_data = [[f"Step {i}", "Unknown", "Unknown"] for i in indices]
            
        fig.add_trace(go.Scatter(
            x=times,
            y=[y_val] * len(times),
            mode='markers',
            marker=dict(size=12, color=color, symbol=symbol, line=dict(width=1, color='black')),
            name=name,
            customdata=custom_data,
            hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Row:</b> %{customdata[0]}<br><b>Content:</b> %{customdata[1]}<br><b>Note Type:</b> %{customdata[2]}<extra></extra>'
        ))

    add_step_trace(indices_perfect, 'Perfect (Beat+Onset)', 'cyan', 2.5)
    add_step_trace(indices_beat, 'Grid (Beat)', 'blue', 2.0)
    add_step_trace(indices_onset, 'Audio (Onset)', 'purple', 1.8)
    add_step_trace(indices_perc, 'Percussive', 'orange', 1.5)
    add_step_trace(indices_bad, 'Unaligned (Bad)', 'red', 1.0, symbol='x')
    add_step_trace(indices_out_of_bounds, 'Out of Bounds (Ignored)', 'gray', 1.0, symbol='circle-x')
    
    # 4. Rhythm Track (Note Type Coloring)
    if results.step_partials:
        rhythm_colors = []
        rhythm_times = []
        rhythm_custom_data = []
        
        color_map = {
            '4th': 'red',
            '8th': 'blue', 
            '12th': 'purple',
            '16th': '#FFD700', # Gold
            '24th': 'pink',
            '32nd': 'orange',
            '48th': 'cyan',
            'Unquantized': 'green'
        }
        
        for i, partial_str in enumerate(results.step_partials):
            # Skip out of bounds for rhythm analysis too
            if results.audio_duration > 0 and step_times[i] > results.audio_duration:
                continue
                
            # Extract key from string like "4th Note (Red)"
            key = partial_str.split(' ')[0] # '4th', '8th'
            color = color_map.get(key, 'gray')
            rhythm_colors.append(color)
            rhythm_times.append(step_times[i])
            
            row_idx = results.step_row_indices[i]
            content = results.step_row_contents[i]
            rhythm_custom_data.append([row_idx, content, partial_str])
            
        fig.add_trace(go.Scatter(
            x=rhythm_times,
            y=[3.5] * len(rhythm_times),
            mode='markers',
            marker=dict(size=14, color=rhythm_colors, symbol='square', line=dict(width=1, color='black')),
            name='Rhythm (Note Type)',
            customdata=rhythm_custom_data,
            hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Note Type:</b> %{customdata[2]}<br><b>Row:</b> %{customdata[0]}<extra></extra>'
        ))

    
    # 5. Beat Placement Track (On vs In Beat) - User Request
    if results.step_partials:
        on_beat_indices = []
        in_beat_indices = []
        
        for i, partial_str in enumerate(results.step_partials):
            # Skip out of bounds
            if results.audio_duration > 0 and step_times[i] > results.audio_duration:
                continue
                
            # "On the Beat" = 4th notes (Red)
            if "4th" in partial_str:
                on_beat_indices.append(i)
            # "In the Beat" = Everything else (8th, 16th, etc.)
            else:
                in_beat_indices.append(i)
        
        
        # Plot "On the Beat" (Green)
        # add_step_trace(on_beat_indices, 'On the Beat (Main)', '#00CC00', 3.0, symbol='square') 
        # Manual trace for custom hover
        if on_beat_indices:
            fig.add_trace(go.Scatter(
                x=step_times[on_beat_indices],
                y=[3.0] * len(on_beat_indices),
                mode='markers',
                marker=dict(size=12, color='#00CC00', symbol='square', line=dict(width=1, color='black')),
                name='On the Beat',
                hovertemplate='<b>ON THE BEAT</b><br>4th Note (Red)<br>Time: %{x:.3f}s<extra></extra>'
            ))
        
        # Plot "In the Beat" (Magenta)
        if in_beat_indices:
            fig.add_trace(go.Scatter(
                x=step_times[in_beat_indices],
                y=[3.0] * len(in_beat_indices),
                mode='markers',
                marker=dict(size=12, color='#FF00FF', symbol='diamond', line=dict(width=1, color='black')),
                name='In the Beat',
                hovertemplate='<b>IN THE BEAT</b><br>Subdivision (8th/16th)<br>Time: %{x:.3f}s<extra></extra>'
            ))

    # 6. Beat/Measure Grid Lines
    # Calculate beat times based on BPM
    max_plot_time = step_times[-1] + 2.0
    
    if results.audio_duration > 0:
        # Add End of Audio Line
        fig.add_vline(
            x=results.audio_duration, 
            line_width=3, 
            line_dash="dash", 
            line_color="red",
            annotation_text="End of Audio", 
            annotation_position="top right"
        )
        
        # Add Valid Region background (Greenish tint)
        # fig.add_vrect(
        #     x0=0, x1=results.audio_duration,
        #     fillcolor="green", opacity=0.05,
        #     layer="below", line_width=0,
        # )
        
        # Add Invalid Region background (Red tint)
        fig.add_vrect(
            x0=results.audio_duration, x1=max(max_plot_time, results.audio_duration + 10),
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Region after Audio (Invalid)", 
            annotation_position="top left"
        )
        
        # Clamp initial view to audio duration
        initial_range_max = results.audio_duration + 2.0
    else:
        initial_range_max = max_plot_time

    if results.bpm > 0:
        spb = 60.0 / results.bpm
        # Draw grid lines
        current_beat = 0
        current_time = results.offset
        
        while current_time < max_plot_time:
            if current_time >= -0.1: # Don't plot too far back
                is_measure = (current_beat % 4 == 0)
                
                # Plot line
                fig.add_shape(
                    type="line",
                    x0=current_time, y0=0.5, x1=current_time, y1=4.5,
                    line=dict(
                        color="black" if is_measure else "gray",
                        width=2 if is_measure else 1,
                        dash="solid" if is_measure else "dot"
                    ),
                    layer="below"
                )
            
            current_time += spb
            current_beat += 1
            
    # Layout Configuration
    fig.update_layout(
        title=f'Interactive Step Alignment - {os.path.basename(results.beatmap_file)}<br><sup>Row 3.0 shows Placement: Green = ON the Beat, Magenta = IN the Beat.</sup>',
        xaxis_title='Time (seconds)',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 1.5, 1.8, 2, 2.5, 3, 3.5, 4],
            ticktext=['Bad', 'Perc', 'Onset', 'Beat', 'Perfect', 'Placement', 'Rhythm', 'Onsets'],
            range=[0.5, 4.8],
            fixedrange=True # Prevent vertical zooming to keep categories stable
        ),
        xaxis=dict(
            # Dynamic ticks for major grid
            tickmode='auto', 
            gridcolor='lightgray',
            zeroline=False,
            range=[-1, initial_range_max], # Set initial view range
            # Minor grid lines every 50ms (0.05s)
            minor=dict(
                ticklen=4,
                dtick=0.05,  
                gridcolor='rgba(220, 220, 220, 0.5)',
                showgrid=True
            ),
            rangeslider=dict(visible=True) # Add slider for easier navigation
        ),
        hovermode='closest',
        template='plotly_white',
        height=700, # Taller for better view
        dragmode='zoom' # Default to zoom tool
    )

    fig.write_html(output_path)
    print(f"\nInteractive visualization saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Beatmap Validator Module")
    print("Import this module and use validate_beatmap() or compare_beatmaps()")
    print("\nExample:")
    print("  from beatmap_validator import validate_beatmap, print_validation_report")
    print("  results = validate_beatmap('song.mp3', 'beatmap.txt', bpm=180, offset=-0.028)")
    print("  print_validation_report(results)")
