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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

__version__ = "0.2.2"
__author__ = "Mukesh Guntumadugu" 

_client: genai.Client | None = None

class Beat(BaseModel):
    time: float = Field(..., description="Timestamp of the beat in seconds")
    note: str = Field(..., description="Type of note (e.g., 'up','down','left','right',etc)")
    thinking: str = Field(..., description=" why up are down why not others ?")

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

@retry(
    retry=retry_if_exception_type((ServerError, ClientError)),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=120)
)
def generate_content_with_retry(prompt, audio_file):
    """Helper to retry generation on failure."""
    try:
        return _client.models.generate_content(
            model="gemini-flash-latest",
            contents=[prompt, audio_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Beat]
            )
        )
    except ClientError as e:
        # Only retry on 429
        if e.code == 429:
             print("Hit 429 (Resource Exhausted). Retrying...")
             raise e
        else:
             # Re-raise other client errors immediately (e.g. 400 Bad Request)
             raise e

def generate_beatmap_chunk(slice_path: str, prompt: str = None) -> list[Beat]:
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
        response = generate_content_with_retry(prompt, audio_file)

        # Parse the response text into our Pydantic models
        if not response.text:
            return []
        
        # The SDK might return parsed object or we might need to parse JSON. 
        # With response_schema, response.parsed might be available? 
        # But safest is json.loads(response.text)
        data = json.loads(response.text)
        return [Beat(**item) for item in data]

    except Exception as e:
        print(f"Error processing slice {slice_path}: {e}")
        return []

def process_full_song(audio_path: str, level: str = "Hard"):
    """
    Processes the entire song in 10s chunks and saves to CSV.
    """
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    # Get total duration
    total_duration = librosa.get_duration(path=audio_path)
    print(f"Processing '{audio_path}' (Duration: {total_duration:.2f}s)")
    
    chunk_duration = 10.0
    all_beats = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_slice_path = os.path.join(script_dir, "temp_current_slice.ogg")

    try:
        num_chunks = math.ceil(total_duration / chunk_duration)
        
        for i in range(num_chunks):
            offset = i * chunk_duration
            print(f"Processing chunk {i+1}/{num_chunks} (Time: {offset:.1f}s - {min(offset + chunk_duration, total_duration):.1f}s)...")
            
            has_audio = create_audio_slice(audio_path, temp_slice_path, offset, chunk_duration)
            if not has_audio:
                break
            
            chunk_beats = generate_beatmap_chunk(temp_slice_path)
            
            # Adjust timestamps
            for beat in chunk_beats:
                beat.time += offset # Add offset to make time relative to start of song
                all_beats.append(beat)
                
            # Sleep briefly to be nice to the API/local checks
            time.sleep(5) 

        # Generate CSV Filename
        # Format: {OriginalName}_{Timestamp}_{Level}.csv
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{original_name}_{timestamp}_{level}.csv"
        csv_path = os.path.join(script_dir, csv_filename)
        
        print(f"Saving to CSV: {csv_path}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['time', 'note', 'thinking']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for beat in all_beats:
                writer.writerow(beat.model_dump())
                
        print("Done!")

    finally:
        if os.path.exists(temp_slice_path):
            os.remove(temp_slice_path)

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
            # Process full song and save to CSV
            process_full_song(test_audio, level="Hard")
        else:
            print(f"Test audio file not found: {test_audio}")








