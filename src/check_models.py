import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not set. ") 
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing supported models...")
try:
    models = client.models.list()
    for m in models:
        pass
        # print(f"Name: {m.name}") # Commenting to reduce noise
    print("✅ Gemini API connected successfully")
except Exception as e:
    print(f"❌ Error listing Gemini models: {e}")

print("\nChecking Qwen local setup...")
try:
    from src.qwen_interface import DEFAULT_MODEL_ID
    print(f"✅ Qwen interface is available. Default model: {DEFAULT_MODEL_ID}")
    print("   (Run `setup_qwen()` in qwen_interface to fully load the transformers model into VRAM)")
except ImportError:
    print("❌ Error: qwen_interface could not be imported")
