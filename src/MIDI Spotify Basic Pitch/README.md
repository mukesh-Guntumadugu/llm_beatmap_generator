# MIDI Conversion with Spotify Basic-Pitch

This module uses **Spotify's [Basic-Pitch](https://github.com/spotify/basic-pitch)** — a free, open-source neural network — to convert raw audio files (MP3, OGG, WAV, etc.) directly into MIDI data. No cloud API, no cost, runs fully locally.

---

## File Structure

```
src/MIDI Spotify Basic Pitch/
├── README.md          ← This file
└── audio_to_midi.py   ← Main conversion script
```

Output is saved in a `midi/` folder **inside each song's directory**:
```
src/musicForBeatmap/Fraxtil's Arrow Arrangements/<SongName>/
├── <SongName>.ogg
├── <SongName>.ssc
├── midi/                          ← Basic-Pitch outputs go here
│   ├── <SongName>_basic_pitch.mid
│   └── <SongName>_basic_pitch.csv
├── qwen_outputs/                  ← Qwen beatmap outputs
└── gemini_outputs/                ← Gemini beatmap outputs (future)
```

---

## Installation (Do these once)

### Step 1 — Install basic-pitch with ONNX backend
```bash
pip install 'basic-pitch[onnx]'
```

### Step 2 — Upgrade resampy (fixes Python 3.12 compatibility)
```bash
pip install --upgrade resampy
```

### Step 3 — Patch basic-pitch for newer scipy (one-time manual fix)
`scipy 1.11+` removed `scipy.signal.gaussian`. basic-pitch 0.3.0 still uses it.
Open the file below and change line 193:

**File to edit:**
```
~/.pyenv/versions/3.12.2/lib/python3.12/site-packages/basic_pitch/note_creation.py
```

**Find (line 193):**
```python
freq_gaussian = scipy.signal.gaussian(window_length, std=5)
```

**Replace with:**
```python
freq_gaussian = scipy.signal.windows.gaussian(window_length, std=5)
```

> This patch is needed because scipy dropped `scipy.signal.gaussian` in 1.11 and basic-pitch 0.3.0 hasn't been updated yet. The fix moves it to the correct new location `scipy.signal.windows.gaussian`.

---

## Running

> All commands below assume you are inside the `src/MIDI Spotify Basic Pitch/` folder:
> ```bash
> cd "src/MIDI Spotify Basic Pitch"
> ```

### Convert a single song
```bash
python audio_to_midi.py "../musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"
```

### Convert ALL songs in the dataset (batch mode)
```bash
python audio_to_midi.py "../musicForBeatmap/" --batch
```

### Filter by frequency range (isolate instruments)
```bash
# Bass only (40–300 Hz)
python audio_to_midi.py "../musicForBeatmap/" --batch --min-freq 40 --max-freq 300

# Melody only (300–2000 Hz)
python audio_to_midi.py "../musicForBeatmap/" --batch --min-freq 300 --max-freq 2000
```

### Change note sensitivity (lower = more notes detected)
```bash
python audio_to_midi.py "song.ogg" --onset-threshold 0.3
```

### Specify a custom output directory
```bash
python audio_to_midi.py "song.ogg" --out-dir /path/to/output/
```

---

## Output Files

For each audio file, two files are saved inside the song's `midi/` folder:

| File | Description |
|---|---|
| `<Name>_basic_pitch.mid` | Standard MIDI — open in GarageBand, Logic, Ableton, etc. |
| `<Name>_basic_pitch.csv` | Note events: `start_s, end_s, pitch_midi, velocity, pitch_bend` |

### Example CSV
```
start_s,end_s,pitch_midi,velocity,pitch_bend
0.0000,0.2344,52,89,0
0.2344,0.4688,55,76,0
0.4688,0.9375,57,92,0
```
`pitch_midi` = standard MIDI note number: `60=C4`, `69=A4 (440Hz)`.

---

## CLI Reference

```
python audio_to_midi.py [input] [options]

Arguments:
  input                    Audio file or directory path

Options:
  --out-dir DIR            Output directory (default: midi/ next to audio file)
  --batch                  Process all audio files in input directory recursively
  --min-freq FLOAT         Minimum frequency in Hz (filters lower pitches)
  --max-freq FLOAT         Maximum frequency in Hz (filters higher pitches)
  --onset-threshold FLOAT  Note sensitivity 0.0–1.0 (default: 0.5, lower = more notes)
  --no-csv                 Skip saving the note events CSV
```

---

## Packages Required

| Package | Why needed | Install |
|---|---|---|
| `basic-pitch` | Spotify's audio-to-MIDI neural network | `pip install 'basic-pitch[onnx]'` |
| `onnxruntime` | Model inference backend (included with above) | included |
| `resampy>=0.4.3` | Audio resampling (fixes `pkg_resources` error on Python 3.12) | `pip install --upgrade resampy` |
| `librosa` | Audio loading | already installed |
| `scipy` | Signal processing (patch note_creation.py — see above) | already installed |

---

## Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named 'pkg_resources'` | Old `resampy 0.4.2` uses deprecated API | `pip install --upgrade resampy` |
| `scipy.signal has no attribute 'gaussian'` | `scipy 1.11+` removed this function | Patch `note_creation.py` line 193 (see above) |
| `'_UserObject' has no attribute 'add_slot'` | TF 2.16 can't load the TF SavedModel | Use `--model-serialization onnx` (already set in script) |
| File already exists, skipping | basic-pitch won't overwrite by default | Delete the `midi/` folder and re-run |

---

## Integration with Beatmap Pipeline

The `<Name>_basic_pitch.csv` note events can feed directly into the Qwen/Gemini beatmap pipeline as an alternative onset source — replacing raw `librosa` onset detection with precise MIDI note start times for better musical accuracy.
