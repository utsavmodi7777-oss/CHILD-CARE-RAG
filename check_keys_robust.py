
import os
import sys
import openai
import cohere
from tavily import TavilyClient

def clean_value(val):
    if not val: return ""
    return val.strip().replace('\x00', '')

def load_env_manual():
    env_vars = {}
    try:
        with open('.env', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    env_vars[key.strip()] = clean_value(val)
    except Exception as e:
        print(f"Error reading .env: {e}")
    return env_vars

def check_openai(key):
    if not key:
        print("❌ OpenAI API Key missing")
        return False
    try:
        client = openai.OpenAI(api_key=key)
        client.models.list()
        print("✅ OpenAI API Key is valid")
        return True
    except Exception as e:
        print(f"❌ OpenAI API Key invalid or error: {str(e)[:100]}...")
        return False

def check_cohere(key):
    if not key:
        print("❌ Cohere API Key missing")
        return False
    try:
        co = cohere.Client(key)
        # Using tokenize as a cheap call to check auth
        co.tokenize(text="test", model="command") 
        print("✅ Cohere API Key is valid")
        return True
    except Exception as e:
        print(f"❌ Cohere API Key invalid or error: {str(e)[:100]}...")
        return False

def check_tavily(key):
    if not key:
        print("❌ Tavily API Key missing")
        return False
    try:
        tavily = TavilyClient(api_key=key)
        # Simple search to test auth
        tavily.search(query="test", max_results=1)
        print("✅ Tavily API Key is valid")
        return True
    except Exception as e:
        print(f"❌ Tavily API Key invalid or error: {str(e)[:100]}...")
        return False

if __name__ == "__main__":
    print("Checking API Keys (Manual Load)...")
    env = load_env_manual()
    
    o = check_openai(env.get("OPENAI_API_KEY"))
    c = check_cohere(env.get("COHERE_API_KEY"))
    t = check_tavily(env.get("TAVILY_API_KEY"))
    
    if o and c and t:
        print("\nAll keys appear valid.")
    else:
        print("\nSome keys are invalid or missing.")
