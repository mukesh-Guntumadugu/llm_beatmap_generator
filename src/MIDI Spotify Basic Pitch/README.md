# MIDI Conversion with Spotify Basic-Pitch

This module uses **Spotify's [Basic-Pitch](https://github.com/spotify/basic-pitch)** — a free, open-source neural network — to convert raw audio files (MP3, OGG, WAV, etc.) directly into MIDI data. No cloud API, no cost, runs fully locally.

---

## What is Basic-Pitch?

Basic-Pitch is a lightweight audio-to-MIDI model trained by Spotify Research. It detects:
- **Note onsets** (when notes start)
- **Note pitch** (what note it is, in MIDI pitch 0–127)
- **Note duration** (how long each note sustains)
- **Pitch bends** (subtle pitch variation within notes)

It works on any polyphonic audio — meaning it can handle chords, melody + bass at the same time.

---

## Installation

```bash
pip install basic-pitch
```

> Basic-Pitch supports Python 3.7–3.11 on Mac, Windows, and Linux.
> For Mac M1, use Python 3.10 specifically.

---

## File Structure

```
src/MIDI Spotify Basic Pitch/
├── README.md          ← This file
└── audio_to_midi.py   ← Main conversion script
```

Output is saved in a `midi_output/` folder next to your audio file (or wherever you specify with `--out-dir`).

---

## Usage

### Convert a single file

```bash
cd src/MIDI\ Spotify\ Basic\ Pitch/
python audio_to_midi.py path/to/song.mp3
```

### Convert all songs in a directory (batch mode)

```bash
python audio_to_midi.py path/to/musicForBeatmap/ --batch
```

### Filter by frequency range (useful for isolating instruments)

```bash
# Bass only (40–300 Hz)
python audio_to_midi.py song.mp3 --min-freq 40 --max-freq 300

# Melody only (300–2000 Hz)
python audio_to_midi.py song.mp3 --min-freq 300 --max-freq 2000
```

### Change note sensitivity (lower = more notes detected)

```bash
python audio_to_midi.py song.mp3 --onset-threshold 0.3
```

---

## Output Files

For each audio file, the script saves:

| File | Description |
|---|---|
| `SongName.mid` | Standard MIDI file — open in any DAW or MIDI editor |
| `SongName_note_events.csv` | CSV with columns: `start_s, end_s, pitch_midi, velocity, pitch_bend` |

---

## Example Output CSV

```
start_s,end_s,pitch_midi,velocity,pitch_bend
0.0000,0.2344,52,89,0
0.2344,0.4688,55,76,0
0.4688,0.9375,57,92,0
```

`pitch_midi` follows standard MIDI numbering: `60 = C4 (middle C)`, `69 = A4 (440 Hz)`, etc.

---

## CLI Reference

```
python audio_to_midi.py [input] [options]

Arguments:
  input                    Audio file or directory path

Options:
  --out-dir DIR            Output directory (default: midi_output/ next to input)
  --batch                  Process all audio files in input directory recursively
  --min-freq FLOAT         Minimum frequency in Hz (filters out lower pitches)
  --max-freq FLOAT         Maximum frequency in Hz (filters out higher pitches)
  --onset-threshold FLOAT  Note sensitivity 0.0-1.0 (default: 0.5)
  --no-csv                 Skip saving the note events CSV
```

---

## Integration with Beatmap Pipeline

The `note_events.csv` output can be used as an alternative onset source for the Qwen beatmap pipeline — instead of using `librosa` onset detection, you can feed in the precise MIDI note start times for much more musically accurate step placement.
