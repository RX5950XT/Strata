"""On the Colab VM: start strata-dsv41 --serve once per load-time setting, then time requests with per-request keys.

    python bench_ds4.py NAME [engine flags...]      -> appends one JSON line per request to /content/logs/bench.jsonl
"""
import json, os, subprocess, sys, time, threading

sys.path.insert(0, "/content/Strata/tools/dsv41")
sys.path.insert(0, "/content/Strata/tools")
import smoke  # noqa: E402

EXE, PACK, LOG = "/content/eng/strata-dsv41", "/content/m/maya-ds4", "/content/logs/bench.jsonl"
BOS, U, A = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜></think>"
PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M into total seconds, with tests.",
    "prose": "Explain in about 300 words why the sky is blue and why sunsets are red, for a curious teenager.",
    "zh": "用繁體中文詳細說明 TCP 三次握手與四次揮手的流程，以及為什麼需要 TIME_WAIT。",
}
# per-request variants: MTP (DSpark) off, on with the engine's floor, and two other floors
VARIANTS = [("mtp_off", "spec=0"), ("mtp_on", "spec=1"), ("mtp_c0.5", "spec=1 dspark_conf=0.5"),
            ("mtp_c0.8", "spec=1 dspark_conf=0.8"), ("mtp_cum0.5", "spec=1 dspark_cum=0.5")]
N = int(os.environ.get("BENCH_N", "256"))


def main():
    name, flags = sys.argv[1], sys.argv[2:]
    tok = smoke.tokenizer(__import__("pathlib").Path(PACK))
    env = dict(os.environ, LD_LIBRARY_PATH="/content/eng:" + os.environ.get("LD_LIBRARY_PATH", ""), STRATA_DS4_REUSE="0", STRATA_DS4_SLOT_KEEP="0")
    cmd = [EXE, "--pack", PACK, "--max-context", "204800", "--eos", "1", *flags, "--serve"]
    err = open(f"/content/logs/eng-{name}.log", "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, env=env, bufsize=1)
    info = []
    for line in p.stdout:
        info.append(line.strip())
        if line.startswith("READY"):
            break
    else:
        rec(name, flags, {"error": "engine exited", "rc": p.wait()})
        return
    load_s = time.time() - t0
    rec(name, flags, {"load_s": round(load_s, 1), "info": info[-12:]})

    def gen(ids, n, keys):
        p.stdin.write(f"GEN {n} {keys} {','.join(map(str, ids))}\n")
        p.stdin.flush()
        out = []
        for line in p.stdout:
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                f = line.split()
                return out, {"gen": int(f[1]), "prompt": int(f[2]), "prefill_ms": float(f[3]), "decode_ms": float(f[4]),
                             "acc": int(f[6]), "off": int(f[7])}
            elif line.startswith("ERR"):
                return out, {"error": line.strip()}
        return out, {"error": "engine exited"}

    pick = os.environ.get("BENCH_PICK")
    variants = [v for v in VARIANTS if not pick or v[0] in pick.split(",")]
    # the expert cache warms over the first requests: several rounds, variants interleaved, so order does not decide
    for rnd in range(int(os.environ.get("BENCH_ROUNDS", "3"))):
        for pk, text in PROMPTS.items():
            ids = tok.encode(BOS + U + text + A, parse_special=True)
            for vn, keys in variants:
                out, d = gen(ids, N, "temperature=0 " + keys)
                if "decode_ms" in d:
                    d["tok_s"] = round(1000 * d["gen"] / max(d["decode_ms"], 1), 2)
                d["text"] = tok.decode(out)[:80] if out else ""
                rec(name, flags, {"round": rnd, "prompt_kind": pk, "variant": vn, **d})
    if os.environ.get("BENCH_LONG", "1") == "1":               # prefill speed at 16K and 64K tokens of filler code
        filler = open("/content/Strata/tools/dsv41/pack_gguf.py", encoding="utf-8").read()
        base = tok.encode(filler)
        for L in [int(x) for x in os.environ.get("BENCH_LONG_LENS", "16384,65536").split(",")]:
            ids = tok.encode(BOS + U, parse_special=True) + (base * (L // len(base) + 1))[:L] + \
                tok.encode("\nSummarize what this code does." + A, parse_special=True)
            out, d = gen(ids, 64, "temperature=0 spec=1")
            if "prefill_ms" in d:
                d["prefill_tok_s"] = round(1000 * d["prompt"] / max(d["prefill_ms"], 1), 1)
                d["tok_s"] = round(1000 * d["gen"] / max(d["decode_ms"], 1), 2)
            rec(name, flags, {"prompt_kind": f"long{L}", "variant": "mtp_on", **d})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()


def rec(name, flags, d):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cfg": name, "flags": " ".join(flags), **d}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
