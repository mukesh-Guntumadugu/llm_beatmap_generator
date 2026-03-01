"""
Mistral (Ollama) integration for LLM Beatmap Generator.

Drop-in companion to gemini.py: same public API, same BeatCSV output schema.
Because Mistral is text-only, audio features are extracted locally via librosa
and forwarded to Mistral as a compact structured-text prompt.

Audio pipeline:
  audio file ──► librosa ──► BPM / onset / beat / RMS features
                                │
                                ▼
                        Ollama  (mistral:latest)
                        POST localhost:11434/api/chat
                                │
                                ▼
                        JSON  ──► list[BeatCSV]

Requires:
  • Ollama running locally  (brew install ollama && ollama serve)
  • mistral model pulled    (ollama pull mistral)
  • librosa, soundfile, numpy  (pip install librosa soundfile numpy)

Usage (mirrors gemini.py):
    from src.mistral import process_full_song, generate_beatmap_csv

    rows = generate_beatmap_csv("song.ogg", duration=180.0, difficulty="Hard")
    # or
    process_full_song("song.ogg", level="Hard", mode="chunked")
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

__version__ = "0.1.0"
__author__ = "Mukesh Guntumadugu"

# ── Shared schema (identical to gemini.py so callers need no changes) ────────

class BeatCSV(BaseModel):
    time_ms: float = Field(..., description="Exact timestamp in milliseconds.")
    beat_position: float = Field(..., description="Beat number from song start.")
    notes: str = Field(..., description="4-char StepMania row or ',' separator.")
    placement_type: int = Field(
        ...,
        description="0=unsure 1=onset 2=beat 3=grid 4=percussive 5=unaligned -1=sep",
    )
    note_type: int = Field(
        ...,
        description="0=whole 1=half 2=quarter 3=eighth 4=extended -1=sep",
    )
    confidence: float = Field(..., description="0.0-1.0 confidence.")
    instrument: str = Field(..., description="kick/snare/bass/melody/unknown/separator")


# ── Ollama helpers ───────────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ollama_available() -> bool:
    """Return True if Ollama is reachable."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _chat(messages: list[dict], model: str = "mistral:latest", temperature: float = 0.3) -> str:
    """
    Send a chat request to Ollama and return the assistant message text.
    Uses format:'json' to encourage valid JSON output.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": 8192},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama not reachable at {OLLAMA_BASE}. "
            "Run: ollama serve && ollama pull mistral"
        ) from exc


# ── Audio feature extraction ─────────────────────────────────────────────────

def _extract_features(audio_path: str, offset: float = 0.0, duration: Optional[float] = None) -> dict:
    """
    Extract timing and energy features from an audio segment.

    Returns a dict with:
      bpm, onset_times, beat_times, audio_duration,
      energy_by_16th  (list of RMS values, one per 16th-note slot)
    """
    y, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=duration)
    audio_dur = librosa.get_duration(y=y, sr=sr)

    # BPM + beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # RMS energy per 16th-note slot
    spb = 60.0 / max(bpm, 1)           # seconds per beat
    slot_dur = spb / 4                  # 16th-note duration
    n_slots = max(1, int(audio_dur / slot_dur))

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    energy_by_16th: list[float] = []
    for s in range(n_slots):
        t0, t1 = s * slot_dur, (s + 1) * slot_dur
        mask = (rms_times >= t0) & (rms_times < t1)
        val = float(np.mean(rms[mask])) if mask.any() else 0.0
        energy_by_16th.append(round(val, 4))

    return {
        "bpm": round(bpm, 2),
        "audio_duration": round(audio_dur, 3),
        "onset_times": [round(t, 3) for t in onset_times],
        "beat_times": [round(t, 3) for t in beat_times],
        "energy_by_16th": energy_by_16th,
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a StepMania DDR beatmap generator. You receive musical analysis data \
(BPM, onset times, beat times, RMS energy per 16th-note slot) for an audio segment \
and output a JSON array of step objects.

SCHEMA — each object MUST have exactly these fields:
  time_ms        (float)  – timestamp in milliseconds
  beat_position  (float)  – beat number from start (1.0 = beat 1)
  notes          (str)    – 4-char string "LDUR" e.g. "1000", or "," for measure end
  placement_type (int)    – 1=onset 2=beat 3=grid 4=percussive -1=separator
  note_type      (int)    – 2=quarter 3=eighth 4=sixteenth -1=separator
  confidence     (float)  – 0.0–1.0
  instrument     (str)    – kick/snare/bass/melody/unknown/separator

MEASURE RULES:
• Each measure ends with a separator row: {"notes": ",", ...all other fields -1/separator}
• A measure contains EXACTLY 4, 8, or 16 note rows before the separator.
• 4  rows = quarter-note grid  (slow / sparse)
• 8  rows = eighth-note grid   (moderate)
• 16 rows = sixteenth-note grid (fast / dense) ← prefer this for Hard difficulty
• Fill silent slots with "0000".
• Cover the ENTIRE audio duration — do NOT stop early.
• Output ONLY valid JSON array. No markdown, no explanation.
"""


def _build_user_prompt(features: dict, difficulty: str, chunk_start: float) -> str:
    bpm = features["bpm"]
    dur = features["audio_duration"]
    onsets = features["onset_times"]
    beats = features["beat_times"]
    energy = features["energy_by_16th"]

    # Truncate long lists for token efficiency
    onset_str = json.dumps(onsets[:80])
    beat_str = json.dumps(beats[:80])
    energy_str = json.dumps(energy[:128])

    return (
        f"Difficulty: {difficulty}\n"
        f"Chunk start offset: {chunk_start:.3f}s\n"
        f"Audio duration: {dur:.3f}s\n"
        f"Detected BPM: {bpm}\n"
        f"Onset times (s, first 80): {onset_str}\n"
        f"Beat times  (s, first 80): {beat_str}\n"
        f"RMS energy per 16th-note slot (first 128): {energy_str}\n\n"
        f"Generate the full {difficulty} beatmap JSON array for this {dur:.1f}s segment."
    )


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_response(text: str) -> list[BeatCSV]:
    """
    Parse Ollama response text into a list of BeatCSV objects.
    Handles cases where the model wraps the array in an outer object.
    """
    text = text.strip()

    # Some models return {"rows": [...]} or {"beatmap": [...]}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract the first JSON array from the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                raw = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                print(f"  [mistral] JSON parse failed. Raw snippet: {text[:200]}")
                return []
        else:
            print(f"  [mistral] No JSON array found. Raw snippet: {text[:200]}")
            return []

    # Unwrap outer object if needed
    if isinstance(raw, dict):
        for key in ("rows", "beatmap", "steps", "data", "chart"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            # Last resort: take first list value
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
            else:
                print(f"  [mistral] Unexpected JSON structure: {list(raw.keys())}")
                return []

    rows: list[BeatCSV] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(BeatCSV(**item))
        except Exception:
            pass  # skip malformed rows silently

    return rows


# ── Core generation functions ─────────────────────────────────────────────────

def generate_beatmap_csv(
    audio_path: str,
    duration: float,
    difficulty: str = "Hard",
    model_name: str = "mistral:latest",
    chunk_start: float = 0.0,
) -> list[BeatCSV]:
    """
    Extract audio features from *audio_path* and ask Mistral (via Ollama)
    to generate StepMania rows. Returns a list of BeatCSV objects.

    Args:
        audio_path:  Path to audio file (entire song or a pre-sliced chunk).
        duration:    Length of the audio segment in seconds.
        difficulty:  "Easy", "Medium", "Hard", or "Expert".
        model_name:  Ollama model tag (default "mistral:latest").
        chunk_start: Absolute start time of this segment within the song (for
                     correct beat_position calculation). Set to 0.0 for the
                     full song.
    """
    if not _ollama_available():
        raise ConnectionError(
            "Ollama is not running. Start it with: ollama serve"
        )

    print(f"  [mistral] Extracting audio features from {os.path.basename(audio_path)} ...")
    features = _extract_features(audio_path)

    user_prompt = _build_user_prompt(features, difficulty, chunk_start)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"  [mistral] Querying {model_name} (BPM={features['bpm']}) ...")
    t0 = time.time()
    response_text = _chat(messages, model=model_name)
    elapsed = time.time() - t0
    print(f"  [mistral] Response in {elapsed:.1f}s ({len(response_text)} chars)")

    rows = _parse_response(response_text)
    print(f"  [mistral] Parsed {len(rows)} rows")
    return rows


def generate_beatmap_csv_chunked(
    audio_path: str,
    duration: float,
    difficulty: str = "Hard",
    model_name: str = "mistral:latest",
    chunk_duration: float = 30.0,
) -> list[BeatCSV]:
    """
    Split audio into chunks and generate BeatCSV rows for each chunk,
    then concatenate. Keeps context window manageable for local Mistral.

    Args:
        audio_path:     Path to full song audio file.
        duration:       Total song duration in seconds.
        difficulty:     Difficulty level.
        model_name:     Ollama model tag.
        chunk_duration: Size of each audio slice in seconds (default 30s).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_slice = os.path.join(script_dir, "_mistral_temp_chunk.wav")

    num_chunks = math.ceil(duration / chunk_duration)
    print(f"[mistral] Chunked mode: {num_chunks} chunks × {chunk_duration:.0f}s")

    all_rows: list[BeatCSV] = []

    try:
        for i in range(num_chunks):
            offset = i * chunk_duration
            actual_dur = min(chunk_duration, duration - offset)
            print(f"\n[mistral] Chunk {i+1}/{num_chunks}: {offset:.1f}s – {offset + actual_dur:.1f}s")

            # Slice audio to WAV
            y, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=actual_dur)
            if len(y) == 0:
                print("  Skipping empty chunk.")
                continue
            sf.write(temp_slice, y, sr)

            chunk_rows = generate_beatmap_csv(
                audio_path=temp_slice,
                duration=actual_dur,
                difficulty=difficulty,
                model_name=model_name,
                chunk_start=offset,
            )

            if chunk_rows:
                all_rows.extend(chunk_rows)
                print(f"  ✅ {len(chunk_rows)} rows collected (total so far: {len(all_rows)})")
            else:
                print("  ⚠️  No rows from this chunk.")

    finally:
        if os.path.exists(temp_slice):
            os.remove(temp_slice)

    return all_rows


# ── process_full_song (mirrors gemini.py entry point) ────────────────────────

def process_full_song(
    audio_path: str,
    level: str = "Hard",
    mode: str = "chunked",
    model_name: str = "mistral:latest",
):
    """
    Generate a StepMania beatmap for a full song using local Mistral via Ollama.

    Mirrors the gemini.py process_full_song() signature so callers can swap
    backends without code changes.

    Args:
        audio_path:  Path to audio file.
        level:       Difficulty – "Easy", "Medium", "Hard", "Expert".
        mode:        "chunked" (recommended) or "full" (entire song in one call).
        model_name:  Ollama model tag (default "mistral:latest").

    Output:
        Saves a CSV file next to this script:
          <song_name>_<level>_mistral_<timestamp>.csv
    """
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    total_duration = librosa.get_duration(path=audio_path)
    print(f"[mistral] Processing '{audio_path}'")
    print(f"          Duration={total_duration:.2f}s  level={level}  mode={mode}  model={model_name}")

    if not _ollama_available():
        print("[mistral] ERROR: Ollama is not running. Start it with:  ollama serve")
        return

    # Generate rows
    if mode == "full":
        all_rows = generate_beatmap_csv(
            audio_path=audio_path,
            duration=total_duration,
            difficulty=level,
            model_name=model_name,
            chunk_start=0.0,
        )
    else:  # chunked (default — recommended)
        all_rows = generate_beatmap_csv_chunked(
            audio_path=audio_path,
            duration=total_duration,
            difficulty=level,
            model_name=model_name,
            chunk_duration=30.0,
        )

    if not all_rows:
        print("[mistral] No rows generated.")
        return

    # Save CSV
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{song_name}_{level}_mistral_{timestamp}.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("notes\n")
        for row in all_rows:
            f.write(f"{row.notes}\n")

    print(f"\n[mistral] ✅ Done!  {len(all_rows)} rows → {csv_path}")


# ── Quick smoke-test / standalone usage ──────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.mistral <audio_file> [difficulty] [mode]")
        print("       difficulty: Easy | Medium | Hard | Expert  (default: Hard)")
        print("       mode:       chunked | full                 (default: chunked)")
        sys.exit(1)

    audio = sys.argv[1]
    diff = sys.argv[2] if len(sys.argv) > 2 else "Hard"
    mode = sys.argv[3] if len(sys.argv) > 3 else "chunked"

    process_full_song(audio, level=diff, mode=mode)
