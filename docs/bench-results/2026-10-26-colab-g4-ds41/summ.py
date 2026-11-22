# summary of /content/logs/bench.jsonl: per config and variant, decode tok/s of rounds >= 1 (round 0 warms the cache)
import json, collections, statistics as st
rows = [json.loads(l) for l in open("/content/logs/bench.jsonl")]
agg = collections.defaultdict(list)
for d in rows:
    if "load_s" in d or "error" in d:
        print(d["cfg"], "load_s", d.get("load_s"), d.get("error", ""))
    elif d.get("prompt_kind", "").startswith("long"):
        print(d["cfg"], d["prompt_kind"], "prefill", d.get("prefill_tok_s"), "decode", d.get("tok_s"))
    elif d.get("round", 0) >= 1 and "tok_s" in d:
        agg[(d["cfg"], d["variant"])].append((d["prompt_kind"], d["tok_s"], round(d["acc"] / max(d["off"], 1), 2)))
for (c, v), xs in agg.items():
    print(f"{c:6} {v:11} median {st.median(t for _, t, _ in xs):6.1f}  ", " ".join(f"{k}:{t}({a})" for k, t, a in xs))
