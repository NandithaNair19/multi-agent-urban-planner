"""
Minimal test: just one tiny request to Gemini, nothing else.
Run this to check if the connection itself works, before troubleshooting
the full project.

Run:  python3 test_gemini.py
"""
import os
from google import genai

print("Step 1: Checking API key is set...")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY is not set in this terminal session.")
    print("Run: export GEMINI_API_KEY=\"your_key_here\"")
    exit(1)
print(f"Key found, starts with: {api_key[:8]}...")

print("Step 2: Creating client...")
client = genai.Client(api_key=api_key)

print("Step 3: Sending a tiny test request to Gemini (should take a few seconds)...")
try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Say hello in exactly 5 words.",
    )
    print("SUCCESS! Gemini replied:")
    print(response.text)
except Exception as e:
    print("FAILED with error:")
    print(type(e).__name__, "-", e)
