"""Quick vLLM connectivity test — run from terminal."""
import urllib.request, json, time

URL = "http://localhost:8000/v1/models"

print(f"Testing {URL} ...", flush=True)
t0 = time.time()
try:
    with urllib.request.urlopen(URL, timeout=10) as r:
        data = json.loads(r.read())
        models = [m["id"] for m in data.get("data", [])]
        print(f"✓ Server is UP ({time.time()-t0:.1f}s)")
        print(f"  Models: {models}")
except Exception as e:
    print(f"✗ FAIL ({time.time()-t0:.1f}s) — {type(e).__name__}: {e}")
