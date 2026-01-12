import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing supported models...")
try:
    models = client.models.list()
    for m in models:
        print(f"Name: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
