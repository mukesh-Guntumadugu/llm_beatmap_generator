"""
Gemini integration for LLM Beatmap Generator.

We will connect Gemini llm model and interact with it and get a beatmap
and see whether we can generate a working beatmap or not.
"""
import os
import time
from google import genai
from google.genai import types

__version__ = "0.1.0"
__author__ = "Mukesh Guntumadugu"

_client: genai.Client | None = None

def setup_gemini(api_key: str):
    """Configures the Google GenAI Client."""
    global _client
    _client = genai.Client(api_key=api_key)

def generate_beatmap(audio_path: str, prompt: str = None) -> str:
    """
    Uploads an audio file to Gemini and generates a beatmap/timing response.
    """
    global _client
    if not _client:
        # Try to init from env if not set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            setup_gemini(api_key)
        else:
            raise ValueError("Gemini client not initialized. Call setup_gemini first.")

    if not prompt:
        prompt = (
            "Listen to this audio file deeply. Identify the beats and rhythm. "
            "Please output a list of timestamps (in seconds) where a beat occurs, "
            "along with a suggested note type (e.g., 'tap') and difficulty. "
            "Format the output as a JSON list of objects with 'time', 'note', 'difficulty'."
        )

    print(f"Uploading audio file: {audio_path}...")
    # The new SDK upload method
    audio_file = _client.files.upload(file=audio_path)
    
    # Wait for the file to be processed
    while audio_file.state == "PROCESSING":
        print("Processing audio file...")
        time.sleep(2)
        audio_file = _client.files.get(name=audio_file.name)

    if audio_file.state == "FAILED":
        raise ValueError("Audio file processing failed.")
    
    print("Audio file ready. Generating beatmap...")
    
    response = _client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[prompt, audio_file]
    )
    
    return response.text

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example usage for testing
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        setup_gemini(api_key)
        # Test file path - replace with a real file to test
        test_audio = "test_audio.mp3" 
        if os.path.exists(test_audio):
            result = generate_beatmap(test_audio)
            print("Generated Beatmap:")
            print(result)
        else:
            print(f"Test audio file not found: {test_audio}")
