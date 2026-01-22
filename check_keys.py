
import os
from dotenv import load_dotenv
import openai
import cohere
from tavily import TavilyClient
import sys

# Load env vars
load_dotenv()

def check_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API Key missing")
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        print("✅ OpenAI API Key is valid")
        return True
    except Exception as e:
        print(f"❌ OpenAI API Key invalid or error: {e}")
        return False

def check_cohere():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ Cohere API Key missing")
        return False
    
    try:
        co = cohere.Client(api_key)
        co.generate(prompt="Hello", max_tokens=1)
        print("✅ Cohere API Key is valid")
        return True
    except Exception as e:
        print(f"❌ Cohere API Key invalid or error: {e}")
        return False

def check_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ Tavily API Key missing")
        return False
    
    try:
        tavily = TavilyClient(api_key=api_key)
        tavily.search(query="test")
        print("✅ Tavily API Key is valid")
        return True
    except Exception as e:
        print(f"❌ Tavily API Key invalid or error: {e}")
        return False

if __name__ == "__main__":
    print("Checking API Keys...")
    o = check_openai()
    c = check_cohere()
    t = check_tavily()
    
    if o and c and t:
        sys.exit(0)
    else:
        sys.exit(1)
