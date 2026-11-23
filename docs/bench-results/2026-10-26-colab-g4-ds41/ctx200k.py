"""On the Colab VM, against the running server: a ~195K-token prompt with a needle near the start, through /v1/chat/completions.

    python ctx200k.py [target_tokens]
"""
import json, sys, time, urllib.request

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 195000
src = open("/content/Strata/setup.py", encoding="utf-8").read()     # ~4 chars a token for code
needle = "The secret launch code for project Bluefin is 7341-KAPPA."
body = needle + "\n\n" + (src * (TARGET * 4 // len(src) + 1))[: TARGET * 3]
req = {"model": "ds", "max_tokens": 200, "temperature": 0, "reasoning_effort": "none",
       "messages": [{"role": "user", "content": body + "\n\nQuestion: what is the secret launch code for project Bluefin? Answer with the code only."}]}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}), timeout=7200)
d = json.loads(r.read())
print(json.dumps({"seconds": round(time.time() - t0, 1), "usage": d.get("usage"), "answer": d["choices"][0]["message"].get("content"),
                  "timings": d.get("timings")}, ensure_ascii=False))
