"""
Gemini connection test for NakhlaGuard.   (ASCII-only file; needs httpx, already in requirements.txt)
Run it in the SAME PowerShell window where you set the key:
    $env:GEMINI_API_KEY="your_new_key"
    python test_gemini.py
It tells you: is the key set, does Google answer, which models work, and how many seconds each one takes.
Send me the whole output (it never prints your key).
"""
import os, sys, time
import httpx

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("GEMINI_API_KEY is NOT set in this window. Set it first, then run this again."); sys.exit(1)
print("key found, length:", len(key))
H = {"x-goog-api-key": key, "Content-Type": "application/json"}
BASE = "https://generativelanguage.googleapis.com/v1beta"
try:
    r = httpx.get(BASE + "/models?pageSize=200", headers=H, timeout=20)
except Exception as e:
    print("Could not reach Google:", type(e).__name__, "-> network / VPN / firewall problem"); sys.exit(1)
print("list models: status", r.status_code)
if r.status_code != 200:
    print(r.text[:300].replace("\n", " ")); print("-> status 400/401/403 usually means a wrong, deleted or restricted key"); sys.exit(1)
flash = [m["name"].replace("models/", "") for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]]
print("flash models available to this key:", flash[:20])
cands = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"] + [n for n in flash if n not in ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")][:3]
print("\nmodel                                    result")
for m in cands:
    t0 = time.time()
    try:
        r = httpx.post(f"{BASE}/models/{m}:generateContent", headers=H, json={"contents": [{"parts": [{"text": "Answer with one word: OK"}]}]}, timeout=45)
        extra = "" if r.status_code == 200 else "  " + r.text[:110].replace("\n", " ")
        print(f"{m:<40} status {r.status_code}  {time.time() - t0:.1f}s{extra}")
    except Exception as e:
        print(f"{m:<40} ERROR {type(e).__name__}  {time.time() - t0:.1f}s")
print("\nDone.")
