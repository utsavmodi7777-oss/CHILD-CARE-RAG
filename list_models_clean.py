
import os
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Force utf-8 output
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY found")
else:
    genai.configure(api_key=api_key)
    print("Listing available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
