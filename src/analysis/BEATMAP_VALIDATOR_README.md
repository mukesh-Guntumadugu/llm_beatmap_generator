# Beatmap Validator

A comprehensive Python tool for validating beatmap step placements against musical features including onsets, beats, and percussive elements.

## Features

1. **Onset Alignment Validation** - Validates if beatmap steps are placed on detected musical onsets
2. **Beat Alignment Validation** - Checks if steps align with detected beats  
3. **Percussive Feature Analysis** - Validates steps against drums and percussive elements
4. **Comparison Mode** - Compare original beatmaps vs generated beatmaps
5. **Detailed Reporting** - JSON output, text reports, and visualizations

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy librosa matplotlib
```

## Quick Start

### Single Beatmap Validation

```python
from analysis.beatmap_validator import validate_beatmap, print_validation_report

results = validate_beatmap(
    audio_path="path/to/song.mp3",
    beatmap_path="path/to/beatmap.txt",
    bpm=142.0,
    offset=-0.028,
    tolerance_ms=50.0  # 50ms tolerance window
)

print_validation_report(results)
```

### Compare Original vs Generated

```python
from analysis.beatmap_validator import compare_beatmaps, print_comparison_report

original_results, generated_results = compare_beatmaps(
    audio_path="path/to/song.mp3",
    original_beatmap_path="path/to/original.txt",
    generated_beatmap_path="path/to/generated.txt",
    bpm=142.0,
    offset=-0.028
)

print_comparison_report(original_results, generated_results)
```

## Usage

Run the example script:

```bash
cd /path/to/llm_beatmap_generator
python3 src/analysis/validate_beatmap_example.py
```

## Output Files

The validator generates several output files:

- **JSON Results** (`validation_results.json`) - Complete validation metrics in JSON format
- **Visualization** (`validation_visualization.png`) - Timeline view, distance histograms, and alignment summary
- **Console Report** - Human-readable validation report

## Validation Metrics

### 1. Onset Alignment
- **Percentage on Onsets**: % of steps placed within tolerance of detected onsets
- **Mean Distance**: Average distance from steps to nearest onset (in milliseconds)
- **Rating**: Excellent (≥80%), Good (≥60%), Fair (≥40%), Poor (<40%)

### 2. Beat Alignment  
- **Percentage on Beats**: % of steps placed within tolerance of detected beats
- **Mean Distance**: Average distance from steps to nearest beat (in milliseconds)
- **Rating**: Excellent (≥80%), Good (≥60%), Fair (≥40%), Poor (<40%)

### 3. Percussive Alignment
- **Percentage on Percussive**: % of steps aligned with percussive features (drums, etc.)
- **Rating**: Excellent (≥70%), Good (≥50%), Fair (≥30%), Poor (<30%)

## Configuration Parameters

- `audio_path`: Path to audio file (mp3, ogg, wav)
- `beatmap_path`: Path to beatmap file (.txt format with measure separators)
- `bpm`: Beats per minute of the song
- `offset`: Time offset in seconds (from .ssc metadata)
- `tolerance_ms`: Tolerance window for alignment in milliseconds (default: 50ms)

## Example Output

```
============================================================
VALIDATION REPORT
============================================================

📊 OVERVIEW
  Total Steps: 438
  BPM: 142.0
  Offset: -0.028s

🎵 ONSET ALIGNMENT
  Percentage on Onsets: 44.29%
  Steps on Onsets: 194/438
  Mean Distance to Nearest Onset: 3345.48ms
  Rating: ⚠ FAIR

🥁 BEAT ALIGNMENT
  Percentage on Beats: 8.90%
  Steps on Beats: 39/438
  Mean Distance to Nearest Beat: 10226.32ms
  Rating: ❌ POOR

🔊 PERCUSSIVE ALIGNMENT
  Percentage on Percussive Features: 49.32%
  Steps on Percussive: 216/438
  Rating: ⚠ FAIR
```

## API Reference

### Core Functions

#### `validate_beatmap(audio_path, beatmap_path, bpm, offset, tolerance_ms=50.0)`
Validate a single beatmap against audio features.

**Returns**: `ValidationResults` object containing all metrics

#### `compare_beatmaps(audio_path, original_beatmap_path, generated_beatmap_path, bpm, offset, tolerance_ms=50.0)`
Compare two beatmaps (original vs generated).

**Returns**: Tuple of (original_results, generated_results)

### Reporting Functions

#### `print_validation_report(results)`
Print a comprehensive validation report to console.

#### `print_comparison_report(original, generated)`
Print a comparison report between two beatmaps.

#### `save_results_json(results, output_path)`
Save validation results to JSON file.

#### `visualize_validation(results, output_path)`
Create visualization with timeline, histograms, and metric summary.

## Beatmap File Format

Beatmap files should be in text format with measures separated by commas:

```
1000
0100
0000
0010
,
0001
1000
0100
,
```

Where:
- Each line represents a time step
- 4 characters per line: Left, Down, Up, Right
- `0` = no note, `1-4` = note types
- `,` = measure separator

## Technical Details

### Onset Detection
Uses `librosa.onset.onset_detect()` with backtracking for accurate onset detection.

### Beat Tracking
Uses `librosa.beat.beat_track()` for tempo estimation and beat detection.

### Percussive Separation
Uses HPSS (Harmonic-Percussive Source Separation) via `librosa.effects.hpss()` to isolate percussion.

### Alignment Tolerance
Default 50ms tolerance window accounts for:
- Human perception limits (~30ms)  
- Audio processing delays
- Quantization in beatmap timing

## How Alignment Works

### Onset Alignment Explained
The "Onset Alignment" score (e.g., 66%) might seem low even for perfect charts. Here's why:

1.  **Detection is Imperfect**: The algorithm looks for sudden energy spikes (spectral flux). It often misses soft sounds or "flow" sections where steps follow a melody rather than a hard beat.
2.  **False Positives**: The audio might have a ghost note (snare ghost) that triggers a detection, but the chart ignores it.
3.  **Genre Dependency**: In chaotic genres like Breakcore ("Bad Ketchup"), the audio is a wall of noise. The algorithm detects hundreds of onsets, but matching them exactly to the charted 16th notes is statistically difficult.

**Verdict**: If your **Beat Alignment** is high (>80%) and **Percussive Alignment** is high (>70%), your chart is excellent, even if Onset Alignment is lower (60-70%). Rely on Beat Alignment as the primary quality metric.

## Files in this Module

- `beatmap_validator.py` - Core validation module
- `validate_beatmap_example.py` - Example usage script
- `README.md` - This documentation

## License

Part of the LLM Beatmap Generator project.
