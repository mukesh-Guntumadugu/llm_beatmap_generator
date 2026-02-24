"""
Gemini integration for LLM Beatmap Generator.

We will connect Gemini llm model and interact with it and get a beatmap
and see whether we can generate a working beatmap or not.
"""
import os
import time
import json
import csv
import math
import librosa
import soundfile as sf
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
__version__ = "0.2.2"
__author__ = "Mukesh Guntumadugu" 

_client: genai.Client | None = None

class Beat(BaseModel):
    notes: str = Field(..., description="4-character string representing columns Left, Up, Down, Right (e.g. '1000') OR a single comma ',' to separate measures.")

class BeatCSV(BaseModel):
    time_ms: float = Field(..., description="Exact timestamp in milliseconds (decimal precision) when this step occurs.")
    beat_position: float = Field(..., description="Beat number from the start of the song (e.g. 1.0 = beat 1, 1.5 = halfway through beat 1, 2.0 = beat 2). Derived from BPM and time_ms.")
    notes: str = Field(..., description="4-character StepMania row string (Left, Down, Up, Right), e.g. '1000'. Can also be ',' for measure separator.")
    placement_type: int = Field(..., description="Step placement: 0=unsure, 1=onset, 2=beat, 3=grid, 4=percussive, 5=unaligned, -1=separator.")
    note_type: int = Field(..., description="Note duration: 0=whole, 1=half, 2=quarter, 3=eighth, 4=extended, -1=separator.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0 for this placement.")
    instrument: str = Field(..., description="Primary instrument detected (e.g. 'kick', 'snare', 'bass', 'melody', 'guitar', 'synth', 'unknown') or 'separator'.")
def setup_gemini(api_key: str):
    """Configures the Google GenAI Client."""
    global _client
    _client = genai.Client(api_key=api_key)

def check_and_update_usage():
    """Tracks usage timestamps in a local file to prevent creating too many requests."""
    usage_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gemini_usage.json")
    now = time.time()
    usage_data = []

    # Load existing data
    if os.path.exists(usage_file):
        try:
            with open(usage_file, 'r') as f:
                loaded_data = json.load(f)
                # Handle legacy format (list of floats) vs new format (list of dicts)
                if isinstance(loaded_data, list):
                     for item in loaded_data:
                        if isinstance(item, (int, float)):
                            usage_data.append({"timestamp": item, "time": time.strftime("%H:%M:%S", time.localtime(item))})
                        elif isinstance(item, dict) and "timestamp" in item:
                            usage_data.append(item)
        except (json.JSONDecodeError, ValueError):
            usage_data = []

    # Filter timestamps from the last 60 seconds for RATE LIMIT CHECK only
    recent_usage = [entry for entry in usage_data if now - entry["timestamp"] < 60]

    # Check limit (allow max 4 requests per minute)
    if len(recent_usage) >= 4:
        wait_time = 60 - (now - recent_usage[0]["timestamp"]) + 1 # Add 1s buffer
        print(f"Rate limit reached. Waiting for {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        
    # Update log
    new_entry = {
        "timestamp": now,
        "time": time.strftime("%H:%M:%S", time.localtime(now))
    }
    usage_data.append(new_entry)
    
    # Keep only the last 10 entries in the file
    if len(usage_data) > 10:
        usage_data = usage_data[-10:]
    
    try:
        with open(usage_file, 'w') as f:
            json.dump(usage_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save usage log: {e}")

def create_audio_slice(input_path: str, output_path: str, offset: float, duration: float):
    """
    Loads a specific slice of the audio file.
    """
    try:
        y, sr = librosa.load(input_path, sr=None, offset=offset, duration=duration)
        if len(y) == 0:
            return False
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"Failed to slice audio at {offset}s: {e}")
        return False

def is_retryable_error(exception):
    """Custom retry predicate: retry on ServerError or 429."""
    if isinstance(exception, ServerError):
        return True
    if isinstance(exception, ClientError):
        if exception.code == 429:
            print(f"Rate limit hit (429). Retrying...")
            return True
        print(f"Non-retryable ClientError: {exception.code} - {exception.message}")
    return False

@retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=120)
)
def generate_content_with_retry(prompt, audio_file, model_name="gemini-pro-latest"):
    """Helper to retry generation on failure."""
    # Note: exception handling is done by the retry decorator's predicate.
    return _client.models.generate_content(
        model=model_name, 
        contents=[prompt, audio_file],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Beat],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                )
            ]
        )
    )

def generate_beatmap_chunk(slice_path: str, prompt: str = None, model_name: str = "gemini-pro-latest") -> list[Beat]:
    """
    Uploads an audio file to Gemini and generates a beatmap/timing response for that chunk.
    """
    check_and_update_usage()
    
    global _client
    if not _client:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            setup_gemini(api_key)
        else:
            raise ValueError("Gemini client not initialized. Call setup_gemini first.")

    if not prompt:
        prompt = "Listen to the audio and identify the beats and rhythm."

    try:
        # print(f"Uploading audio slice: {slice_path}...")
        audio_file = _client.files.upload(file=slice_path)
        
        while audio_file.state == "PROCESSING":
            time.sleep(1)
            audio_file = _client.files.get(name=audio_file.name)

        # print("Generating beatmap for slice...")
        response = generate_content_with_retry(prompt, audio_file, model_name=model_name)

        # Parse the response text into our Pydantic models
        if not response.text:
            print(f"DEBUG: Empty response text. Candidates: {response.candidates}")
            return []
        
        # print(f"DEBUG: Response text: {response.text[:200]}...") # Optional debug
        
        # The SDK might return parsed object or we might need to parse JSON. 
        # With response_schema, response.parsed might be available? 
        # But safest is json.loads(response.text)
        data = json.loads(response.text)
        return [Beat(**item) for item in data]

    except Exception as e:
        print(f"Error processing slice {slice_path}: {e}")
        return []

# ── Static system instruction (cacheable) ────────────────────────────────────
_BEATMAP_SYSTEM_INSTRUCTION = (
    "You are a StepMania beatmap generator. Output a JSON array of objects.\n"
    "Each object has these fields:\n"
    "  - time_ms (float): exact timestamp in milliseconds\n"
    "  - beat_position (float): beat number from song start\n"
    "  - notes (str): 4-character row (Left, Down, Up, Right e.g. '1000') OR ',' for measure end\n"
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
    "=== EXAMPLE (16-row measure at ~120 BPM) ===\n"
    '[{"time_ms":0.0,"beat_position":1.0,"notes":"1000","placement_type":4,"note_type":2,"confidence":0.95,"instrument":"kick"},\n'
    ' {"time_ms":125.0,"beat_position":1.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":250.0,"beat_position":1.5,"notes":"0010","placement_type":4,"note_type":3,"confidence":0.88,"instrument":"snare"},\n'
    ' {"time_ms":375.0,"beat_position":1.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":500.0,"beat_position":2.0,"notes":"0000","placement_type":0,"note_type":2,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":625.0,"beat_position":2.25,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":750.0,"beat_position":2.5,"notes":"0100","placement_type":4,"note_type":3,"confidence":0.82,"instrument":"snare"},\n'
    ' {"time_ms":875.0,"beat_position":2.75,"notes":"0000","placement_type":0,"note_type":3,"confidence":1.0,"instrument":"unknown"},\n'
    ' {"time_ms":1000.0,"beat_position":3.0,"notes":"1001","placement_type":4,"note_type":2,"confidence":0.91,"instrument":"kick"},\n'
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
    "- Cover the ENTIRE audio from start to finish. Do NOT stop early.\n"
    "- beat_position must be consistent with the detected BPM and time_ms.\n"
)

def create_beatmap_prompt_cache(difficulty: str = "Hard", model_name: str = "gemini-2.0-flash-001", ttl_seconds: int = 3600) -> str | None:
    """
    Creates a Gemini context cache for the static beatmap system instruction.
    Returns the cache name to reuse across all songs, or None if caching fails.
    Cache is valid for ttl_seconds (default 1 hour = entire batch run).
    NOTE: Requires gemini-2.0-flash-001 or gemini-1.5-pro — caching not supported on all models.
    """
    global _client
    if not _client:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            setup_gemini(api_key)
        else:
            raise ValueError("Gemini client not initialized.")

    try:
        print(f"Creating context cache (TTL={ttl_seconds}s)...")
        cached = _client.caches.create(
            model=model_name,
            config={
                "system_instruction": _BEATMAP_SYSTEM_INSTRUCTION + f"\nDifficulty: {difficulty}",
                "ttl": f"{ttl_seconds}s",
            }
        )
        print(f"Cache created: {cached.name}")
        return cached.name
    except Exception as e:
        print(f"Context cache creation failed (will use standard prompt): {e}")
        return None

def generate_beatmap_csv(
    audio_path: str,
    duration: float,
    difficulty: str = "Beginner",  #
    model_name: str = "gemini-2.0-flash-001",
    cached_content_name: str | None = None
) -> list[BeatCSV]:
    """
    Uploads audio to Gemini and returns structured CSV beatmap rows.
    If cached_content_name is provided (from create_beatmap_prompt_cache),
    the static prompt is served from cache — only the audio + short per-song
    instruction is sent each call.
    """
    check_and_update_usage()

    global _client
    if not _client:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            setup_gemini(api_key)
        else:
            raise ValueError("Gemini client not initialized. Call setup_gemini first.")

    # Per-song dynamic part (short — audio duration changes per song)
    per_song_prompt = (
        f"The audio is {duration:.1f} seconds long. "
        f"Generate a {difficulty} difficulty StepMania beatmap for the ENTIRE duration."
    )

    try:
        # Upload audio
        audio_file = _client.files.upload(file=audio_path)
        while audio_file.state == "PROCESSING":
            time.sleep(1)
            audio_file = _client.files.get(name=audio_file.name)

        # Build config — use cache if available, else embed full prompt inline
        if cached_content_name:
            # Cache hit: only send audio + short per-song prompt
            config = types.GenerateContentConfig(
                cached_content=cached_content_name,
                response_mime_type="application/json",
                response_schema=list[BeatCSV],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                ]
            )
            contents = [per_song_prompt, audio_file]
        else:
            # No cache: send full prompt + audio (fallback)
            full_prompt = _BEATMAP_SYSTEM_INSTRUCTION + f"\nDifficulty: {difficulty}\n\n" + per_song_prompt
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[BeatCSV],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                ]
            )
            contents = [full_prompt, audio_file]

        response = _client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        if not response.text:
            print("Empty response from Gemini.")
            return []

        data = json.loads(response.text)
        return [BeatCSV(**item) for item in data]

    except Exception as e:
        err_str = str(e)
        # If the cache expired / is invalid, retry once with the full inline prompt
        if cached_content_name and ("403" in err_str or "PERMISSION_DENIED" in err_str or "CachedContent" in err_str):
            print(f"  Cache miss/expired for {os.path.basename(audio_path)}, retrying with full prompt...")
            try:
                full_prompt = _BEATMAP_SYSTEM_INSTRUCTION + f"\nDifficulty: {difficulty}\n\n" + per_song_prompt
                config_full = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[BeatCSV],
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    ]
                )
                response2 = _client.models.generate_content(
                    model=model_name,
                    contents=[full_prompt, audio_file],
                    config=config_full
                )
                if not response2.text:
                    print("Empty response on retry.")
                    return []
                data = json.loads(response2.text)
                return [BeatCSV(**item) for item in data]
            except Exception as e2:
                print(f"Error in generate_beatmap_csv (retry) for {audio_path}: {e2}")
                return []
        print(f"Error in generate_beatmap_csv for {audio_path}: {e}")
        return []


def process_full_song(audio_path: str, level: str = "Beginner", mode: str = "full"):
    """
    Processes the song based on the selected mode:
    - 'full': Processes the entire song in one go/request to Gemini.
    - 'split': Slices the audio into chunks and processes them sequentially.
    """
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    # Get total duration
    total_duration = librosa.get_duration(path=audio_path)
    print(f"Processing '{audio_path}' (Duration: {total_duration:.2f}s) in '{mode}' mode")
    
    expected_commas = int(total_duration)
    
    # Select prompt based on level
    if level.lower() == "Hard":
        prompt_text = (
            f"The audio is {total_duration:.1f} seconds long. You MUST generate chart data for the ENTIRE duration.\n"
            f"Target: approximately {expected_commas} measure separators (commas) — one for roughly every second of audio.\n\n"
            "Listen to the audio and generate StepMania chart rows for a beginner difficulty. "
            "Output a continuous sequence of 4-character strings covering the entire audio duration. "
            "Each string represents a row in the chart (Left, Up, Down, Right). "
            "Use '0000' for empty rows to maintain correct timing and rhythm (e.g., 4 rows per beat). "
            "IMPORTANT: Separate measures with a comma ',' on its own line/entry. A measure usually has 4 beats.\n"
            "Use the following note codes:\n"
            "0: Empty\n"
            "1: Tap\n"
            "2: Hold Head\n"
            "3: Hold End\n"
            "4: Roll Head\n"
            ",: Measure Separator\n\n"
            "Example Sequence:\n"
            "1000\n"
            "0000\n"
            "0000\n"
            "0000\n"
            ",\n"
            "0010\n"
            "...\n"
            "Focus on strong downbeats using mostly taps. Ensure commas separate logical musical chunks.\n"
            ""
            "DO NOT STOP EARLY. GENERATE UNTIL THE END OF THE SONG."
        )
    else:
        prompt_text = (
            f"The audio is {total_duration:.1f} seconds long. You MUST generate chart data for the ENTIRE duration.\n"
            f"Target: approximately {expected_commas} measure separators (commas) — one for roughly every second of audio.\n\n"
            "Listen to the audio and generate StepMania chart rows for a hard difficulty. "
            "Output a continuous sequence of 4-character strings covering the entire audio duration. "
            "Each string represents a row in the chart (Left, Up, Down, Right). "
            "Use '0000' for empty rows to maintain correct timing and rhythm (e.g., 4 rows per beat). "
            "IMPORTANT: Separate measures with a comma ',' on its own line/entry. A measure usually has 4 beats.\n"
            "Use the following note codes:\n"
            "0: Empty\n"
            "1: Tap\n"
            "2: Hold Head\n"
            "3: Hold End\n"
            "4: Roll Head\n"
            ",: Measure Separator\n\n"
            "Example Sequence:\n"
            "1000\n"
            "0200\n"
            ",\n"
            "0301\n"
            "...\n"
            "Match the intensity of the music with complex patterns. Ensure commas separate logical musical chunks.\n"
            "DO NOT STOP EARLY. GENERATE UNTIL THE END OF THE SONG."
        )

    all_beats = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Check if we should try caching (only for full mode and sufficient duration/tokens)
        # 32k tokens roughly corresponds to > 2 minutes of audio + text.
        # We'll just attempt it for 'full' mode if requested or by default if long enough.
        # For this implementation, we will try standard upload first, or cache if user requested (implied context).
        # Let's add a simple toggle for now or just try-except the cache creation.
        
        if mode == "split":
            chunk_duration = 100.0
            temp_slice_path = os.path.join(script_dir, "temp_current_slice.wav")
            num_chunks = math.ceil(total_duration / chunk_duration)
            
            try:
                for i in range(num_chunks):
                    offset = i * chunk_duration
                    print(f"Processing chunk {i+1}/{num_chunks} (Time: {offset:.1f}s - {min(offset + chunk_duration, total_duration):.1f}s)...")
                    
                    has_audio = create_audio_slice(audio_path, temp_slice_path, offset, chunk_duration)
                    if not has_audio:
                        break
                    
                    chunk_beats = generate_beatmap_chunk(temp_slice_path, prompt=prompt_text)
                    
                    # Store beats directly
                    all_beats.extend(chunk_beats)
                        
                    # Sleep briefly to be nice to the API/local checks
                    time.sleep(5)
            finally:
                 if os.path.exists(temp_slice_path):
                    os.remove(temp_slice_path)

        else: # Full mode
            print(f"Uploading full audio file: {audio_path}...")
            
            check_and_update_usage()
            global _client
            if not _client:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    setup_gemini(api_key)
                else:
                    raise ValueError("Gemini client not initialized.")

            # Attempt caching strategy
            try:
                # First upload the file to File API
                audio_file = _client.files.upload(file=audio_path)
                while audio_file.state == "PROCESSING":
                    time.sleep(1)
                    audio_file = _client.files.get(name=audio_file.name)
                
                print("Audio uploaded. Attempting to create context cache (ttl=10m)...")
                
                # Create cached content
                # Note: This requires the content to be large enough (>32k tokens)
                cache_config = {
                    "contents": [prompt_text, audio_file],
                    "ttl": "600s"
                }
                
                # Check if we are using a pro model for caching benefits
                cached_content = _client.caches.create(model="gemini-2.0-flash-001", config=cache_config)
                print(f"Cache created: {cached_content.name}. Generating content...")

                response = _client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=[], # Content is in cache
                    config=types.GenerateContentConfig(
                        cached_content=cached_content.name,
                        response_mime_type="application/json",
                        response_schema=list[Beat],
                    )
                )
                
                if response.text:
                    data = json.loads(response.text)
                    all_beats = [Beat(**item) for item in data]
                else:
                    print("Empty response from cached generation.")

            except Exception as e:
                print(f"Cache creation/usage failed (likely <32k tokens or quota). Falling back to standard generation: {e}")
                # Fallback to standard generation
                # Re-upload or reuse audio_file object if it exists?
                # The 'generate_beatmap_chunk' handles upload internally, so we can just call that
                # but we already uploaded it above. Let's just call generate_content directly if we have the file.
                
                try:
                    # If audio_file was created above, use it
                    # If it wasn't (e.g. upload failed), generate_beatmap_chunk will try again
                    if 'audio_file' in locals() and audio_file:
                        response = generate_content_with_retry(prompt_text, audio_file)
                        if response.text:
                            data = json.loads(response.text)
                            all_beats = [Beat(**item) for item in data]
                    else:
                        all_beats = generate_beatmap_chunk(audio_path, prompt=prompt_text)
                except Exception as inner_e:
                     print(f"Fallback generation failed: {inner_e}")

        
        if not all_beats:
            print("No beats generated.")
            return

        # Generate CSV Filename (keeping .csv extension for now as requested, but format is raw text/custom)
        # Format: {OriginalName}_{Timestamp}_{Level}_{Mode}.csv
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{original_name}_{timestamp}_{level}_{mode}.csv"
        csv_path = os.path.join(script_dir, csv_filename)
        
        print(f"Saving to CSV: {csv_path}")
        
        # Write raw lines to avoid CSV quoting of the comma separator
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("notes\n") # Header
            for beat in all_beats:
                f.write(f"{beat.notes}\n")
                
        print("Done!")                
        print("Done!")

    except Exception as e:
        print(f"Error processing song: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    # Force reload environment variables from .env file
    load_dotenv(override=True)
    
    # Example usage for testing
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        setup_gemini(api_key)
        # Test file path - replace with a real file to test
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_audio = os.path.join(script_dir, "musicForBeatmap", "Goin' Under", "Goin' Under.ogg") 
        
        if os.path.exists(test_audio):
             # Ask user for mode
            try:
               user_input = input("Enter mode ('full' [default] or 'split'): ").strip().lower()
            except EOFError:
               user_input = "full"
            
            mode = "split" if user_input in ["split", "s"] else "full"
            
            # Process full song and save to CSV
            process_full_song(test_audio, level="Hard", mode=mode)
        else:
            print(f"Test audio file not found: {test_audio}")








