"""
Shared beatmap prompt — single source of truth used by both Gemini and Qwen.

Both models receive the SAME system instruction text.
The only difference is how output is enforced:
  - Gemini: response_schema=list[BeatCSV] forces structured JSON (model ignores format instructions)
  - Qwen:   the CSV output section at the end guides the plain-text output
"""

# ── Static system instruction (word-for-word same for Gemini and Qwen) ────────
BEATMAP_SYSTEM_INSTRUCTION = (
    "You are a StepMania beatmap generator. Output a plain CSV format.\n"
    "Each row MUST have exactly these 7 comma-separated fields:\n"
    "  - time_ms (float): exact timestamp in milliseconds\n"
    "  - beat_position (float): beat number from song start\n"
    "  - notes (str): 4-character row (Left, Down, Up, Right e.g. '1000') OR ',' for measure end\n"
    "      NOTE TYPES:\n"
    "        '0' = Empty slot\n"
    "        '1' = Standard tap note\n"
    "        '2' = Hold note HEAD (start of a freeze arrow)\n"
    "        '3' = Hold note TAIL (end of a freeze arrow)\n"
    "  - placement_type (int): 0=unsure, 1=onset, 2=beat, 3=grid, 4=percussive, 5=unaligned, -1=separator\n"
    "  - note_type (int): 0=whole, 1=half, 2=quarter, 3=eighth, 4=extended, -1=separator\n"
    "  - confidence (float): 0.0-1.0\n"
    "  - instrument (str): kick/snare/bass/melody/guitar/synth/unknown or 'separator'\n\n"
    "=== MEASURE STRUCTURE — READ CAREFULLY ===\n"
    "A measure is a group of rows ended by a separator row (notes=',').\n"
    "EACH measure MUST contain EXACTLY 4, 8, 12, or 16 note rows before the ','.\n"
    "  - 4 rows  = quarter note grid   → 1 row per beat (sparse / slow music)\n"
    "  - 8 rows  = eighth note grid    → 2 rows per beat (moderate density)\n"
    "  - 12 rows = triplet grid        → 3 rows per beat (triplet feel)\n"
    "  - 16 rows = sixteenth note grid → 4 rows per beat (dense / fast music, MOST COMMON for Hard)\n"
    "You CAN and SHOULD use 16 rows per measure whenever the music is busy or fast.\n"
    "At 120 BPM, a 16-row measure spans roughly 2 seconds (8 rows ≈ 1 second).\n"
    "For every subdivision slot that has NO note, you MUST output '0000'.\n"
    "NEVER output two consecutive separator rows (notes=',') back to back.\n"
    "NEVER output a separator with zero note rows before it.\n\n"
    "=== HOLD NOTES (FREEZE ARROWS) RULES ===\n"
    "A hold note starts with a '2' (Head) in a specific lane, and MUST end later with a '3' (Tail) in that EXACT SAME lane.\n"
    "Example of holding the Left arrow for 1 beat:\n"
    "  Row 1 (Beat 1.0): '2000' (Start holding left)\n"
    "  Row 2 (Beat 1.5): '0000' (Still holding...) this can be as long as possible\n"
    "  Row 3 (Beat 2.0): '3000' (Release left)\n"
    "CRITICAL: You MUST NOT place any other notes (1, 2, or 3) in a lane while it is currently being held.\n"
    "CRITICAL: Every '2' MUST be followed eventually by a matching '3' in the same column.\n\n"
    "=== EXAMPLE (16-row measure at ~120 BPM) ===\n"
    "time_ms,beat_position,notes,placement_type,note_type,confidence,instrument\n"
    "0.0,1.0,1000,4,2,0.95,kick\n"
    "125.0,1.25,0000,0,3,1.0,unknown\n"
    "250.0,1.5,0010,4,3,0.88,snare\n"
    "375.0,1.75,0000,0,3,1.0,unknown\n"
    "500.0,2.0,2000,4,2,1.0,bass\n"
    "625.0,2.25,0000,0,3,1.0,unknown\n"
    "750.0,2.5,0100,4,3,0.82,snare\n"
    "875.0,2.75,0000,0,3,1.0,unknown\n"
    "1000.0,3.0,3001,4,2,0.91,kick\n"
    "1125.0,3.25,0000,0,3,1.0,unknown\n"
    "1250.0,3.5,0000,0,3,1.0,unknown\n"
    "1375.0,3.75,0000,0,3,1.0,unknown\n"
    "1500.0,4.0,0000,0,2,1.0,unknown\n"
    "1625.0,4.25,0000,0,3,1.0,unknown\n"
    "1750.0,4.5,0010,4,3,0.79,snare\n"
    "1875.0,4.75,0000,0,3,1.0,unknown\n"
    '2000.0,5.0,",",-1,-1,1.0,separator\n\n'
    "=== OTHER RULES ===\n"
    "- Choose 4, 8, 12, or 16 rows per measure based on the rhythmic density of the music.\n"
    "- Cover the audio slice from start to finish. Do NOT stop early.\n"
    "- beat_position must be consistent with the detected BPM and time_ms.\n"
    "- DO NOT ARGUE with instructions or apologize. You MUST output the notes.\n"
)

# ── Per-song dynamic part (same text for both models) ─────────────────────────
def build_per_song_prompt(difficulty: str, duration: float, bpm: float = None) -> str:
    """Returns the short per-song instruction appended after the system instruction."""
    prompt = (
        f"Difficulty: {difficulty}\n"
        f"The audio is {duration:.1f} seconds long.\n"
    )
    if bpm and bpm > 0:
        measure_duration = 240.0 / bpm
        prompt += (
            f"The underlying BPM of the song is roughly {bpm:.1f}. "
            f"Therefore, a single measure (4 beats) lasts exactly {measure_duration:.3f} seconds.\n"
            f"Please ensure your comma ',' separators appear roughly every {measure_duration:.3f} seconds.\n"
        )
    prompt += f"Generate a {difficulty} difficulty StepMania beatmap for this specific {duration:.1f} second audio slice."
    return prompt

# ── Qwen addendum: reinforce the strictly JSON/CSV format ─────────────────────
QWEN_OUTPUT_ADDENDUM = (
    "\n\n=== STRICT OUTPUT RULES ===\n"
    "You are generating a StepMania BEATMAP, NOT a music transcription or MIDI file.\n"
    "DO NOT output any of these MIDI-style fields: 'time', 'note_name', 'velocity', 'pitch', 'duration', 'frequency'.\n"
    "DO NOT include markdown, explanations, apologies, or conversational text.\n"
    "DO NOT write 'Here is your CSV:' or similar phrases.\n"
    "Do NOT stop early. Keep generating rows until time_ms reaches the end of the audio chunk.\n\n"
    "You MUST output a JSON object with a 'rows' array. Each element in 'rows' MUST have EXACTLY these 7 fields:\n"
    '  {"time_ms": <float>, "beat_position": <float>, "notes": "<4-char string>", '
    '"placement_type": <int>, "note_type": <int>, "confidence": <float>, "instrument": "<string>"}\n\n'
    "EXAMPLE of correct output format:\n"
    '{"rows": [\n'
    '  {"time_ms": 0.0, "beat_position": 1.0, "notes": "1000", "placement_type": 4, "note_type": 2, "confidence": 0.95, "instrument": "kick"},\n'
    '  {"time_ms": 125.0, "beat_position": 1.25, "notes": "0000", "placement_type": 0, "note_type": 3, "confidence": 1.0, "instrument": "unknown"},\n'
    '  {"time_ms": 250.0, "beat_position": 1.5, "notes": "0010", "placement_type": 4, "note_type": 3, "confidence": 0.88, "instrument": "snare"},\n'
    '  {"time_ms": 500.0, "beat_position": 2.0, "notes": ",", "placement_type": -1, "note_type": -1, "confidence": 1.0, "instrument": "separator"}\n'
    "]}\n\n"
    "Output the JSON object immediately, starting with '{\"rows\": [':\n"
)

def build_qwen_prompt(difficulty: str, duration: float, bpm: float = None) -> str:
    """Full prompt for Qwen: same system instruction as Gemini + CSV output guide."""
    return (
        BEATMAP_SYSTEM_INSTRUCTION
        + build_per_song_prompt(difficulty, duration, bpm)
        + QWEN_OUTPUT_ADDENDUM
    )
