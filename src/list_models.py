
import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

client = genai.Client(api_key=api_key)

try:
    print("Listing models...")
    for m in client.models.list():
        if "gemini" in m.name:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
