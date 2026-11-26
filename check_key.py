"""Quick diagnostic: run `python check_key.py` to see why Gemini calls fail."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY not found. Put it in the .env file.")

print(f"Key starts with: {key[:4]}...  length: {len(key)}")


def try_mode(label, **kwargs):
    print(f"\n=== {label} ===")
    try:
        client = genai.Client(api_key=key, **kwargs)
    except Exception as exc:
        print("client error:", exc)
        return False
    ok = False
    for model in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]:
        try:
            r = client.models.generate_content(model=model, contents="Say OK")
            print(f"[OK]   {model}: {r.text.strip()[:30]}")
            ok = True
        except Exception as exc:
            print(f"[FAIL] {model}: {' '.join(str(exc).split())[:220]}")
    try:
        e = client.models.embed_content(model="gemini-embedding-001", contents=["test"])
        print(f"[OK]   embeddings: {len(e.embeddings[0].values)} dims")
    except Exception as exc:
        print(f"[FAIL] embeddings: {' '.join(str(exc).split())[:220]}")
    return ok


if not try_mode("Standard Gemini API"):
    try_mode("Vertex express mode", vertexai=True)
