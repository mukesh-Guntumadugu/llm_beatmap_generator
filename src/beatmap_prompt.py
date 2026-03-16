"""
Shared beatmap prompt — single source of truth used by both Gemini and Qwen.

Both models receive the SAME system instruction text.
The only difference is how output is enforced:
  - Gemini: response_schema=list[BeatCSV] forces structured JSON (model ignores format instructions)
  - Qwen:   the CSV output section at the end guides the plain-text output
"""

# ── Static system instruction (word-for-word same for Gemini and Qwen) ────────
BEATMAP_SYSTEM_INSTRUCTION = (
    "You are a StepMania beatmap generator. Output a JSON array of objects.\n"
    "Each object has these fields:\n"
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
    '[{"time_ms":0.0,"beat_position":1.0,"notes":"1000","placement_type":4,"note_type":2,"confidence":0.95,"instrument":"kick"},\n'
    ' {"time_ms":125.0,"beat_position":1.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":250.0,"beat_position":1.5,"notes":"0010","placement_type":4,"note_type":3,"confidence":0.88,"instrument":"snare"},\n'
    ' {"time_ms":375.0,"beat_position":1.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":500.0,"beat_position":2.0,"notes":"2000","placement_type":4,"note_type":2,"confidence":1.0,"instrument":"bass"}, <-- HOLD HEAD (Left)\n'
    ' {"time_ms":625.0,"beat_position":2.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":750.0,"beat_position":2.5,"notes":"0100","placement_type":4,"note_type":3,"confidence":0.82,"instrument":"snare"},\n'
    ' {"time_ms":875.0,"beat_position":2.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1000.0,"beat_position":3.0,"notes":"3001","placement_type":4,"note_type":2,"confidence":0.91,"instrument":"kick"}, <-- HOLD TAIL (Left) + Tap (Right)\n'
    ' {"time_ms":1125.0,"beat_position":3.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1250.0,"beat_position":3.5,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1375.0,"beat_position":3.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1500.0,"beat_position":4.0,"notes":"0000","placement_type":0,"note_type":2,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1625.0,"beat_position":4.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1750.0,"beat_position":4.5,"notes":"0010","placement_type":4,"note_type":3,"confidence":0.79,"instrument":"snare"},\n'
    ' {"time_ms":1875.0,"beat_position":4.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":2000.0,"beat_position":5.0,"notes":",","placement_type":-1,"note_type":-1,"confidence":1.0,"instrument":"separator"}]\n\n'
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

# ── Qwen addendum: since Qwen has no response_schema, guide its output format ─
QWEN_OUTPUT_ADDENDUM = (
    "\n\n=== OUTPUT FORMAT (for this model) ===\n"
    "Output ONLY plain CSV rows (no header, no extra JSON wrapping, no markdown, no explanations).\n"
    "Each line: time_ms,beat_position,notes,placement_type,note_type,confidence,instrument\n"
    'For separator rows use: time_ms,beat_position,",",-1,-1,1.0,separator\n\n'
    "Example output (copy this format exactly):\n"
    "0.0,1.0,1000,4,2,0.95,kick\n"
    "125.0,1.25,0000,0,3,1.0,unknown\n"
    "250.0,1.5,0010,4,3,0.88,snare\n"
    "375.0,1.75,0000,0,3,1.0,unknown\n"
    "500.0,2.0,2000,4,2,1.0,bass\n"
    "625.0,2.25,0000,0,3,1.0,unknown\n"
    "750.0,2.5,0100,4,3,0.82,snare\n"
    "875.0,2.75,0000,0,3,1.0,unknown\n"
    "1000.0,3.0,3001,4,2,0.91,kick\n"
    '1125.0,3.25,",",-1,-1,1.0,separator\n\n'
    "CRITICAL WARNING:\n"
    "DO NOT output ANY conversational text, apologies, or explanations.\n"
    "DO NOT complain about constraints. Just output the best CSV you can.\n"
    "You MUST continue generating lines until `time_ms` reaches the end of this audio chunk.\n"
    "Start immediately with the CSV data:\n"
)

def build_qwen_prompt(difficulty: str, duration: float, bpm: float = None) -> str:
    """Full prompt for Qwen: same system instruction as Gemini + CSV output guide."""
    return (
        BEATMAP_SYSTEM_INSTRUCTION
        + build_per_song_prompt(difficulty, duration, bpm)
        + QWEN_OUTPUT_ADDENDUM
    )
