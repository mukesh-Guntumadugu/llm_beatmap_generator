import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("API Key not found")
else:
    client = genai.Client(api_key=api_key)
    try:
        print("Listing models...")
        for model in client.models.list():
            print(f"- {model.name}")
    except Exception as e:
        print(f"Error: {e}")
