#!/usr/bin/env python3
"""Measure Strata's streaming chat latency across prompt lengths."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import mimetypes
import os
import re
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".md", ".py", ".cpp", ".hpp", ".cu"}
# engine 0.1.2: "N prompt tokens in ..."; 0.1.3+: "prompt N tokens = R reused + M read in ..., drafts accepted A of B"
ENGINE_LINE = re.compile(
    r"strata serve: (?:prompt )?(\d+) (?:prompt )?tokens.*? in ([\d.]+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)"
)
MTP_LINE = re.compile(r"drafts accepted (\d+) of (\d+)")


def cli():
    # argparse sees a value beginning with -- as a new option unless joined with =.
    argv = sys.argv[1:]
    for i in range(len(argv) - 1):
        if argv[i] in ("--arg", "--drop") and argv[i + 1].startswith("--"):
            argv[i] += "=" + argv[i + 1]
            argv[i + 1] = ""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="configs/strata-*.json from setup.py")
    ap.add_argument("--arg", action="append", default=[], metavar="FLAG=VALUE")
    ap.add_argument("--drop", action="append", default=[], metavar="FLAG")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--contexts", default="1024,8192,32768,65536,131072,204800,262144")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--vision-test", type=Path, metavar="IMAGE")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--nonce", help="fixed prompt nonce, so different configs get the same prompts (default: random)")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args([x for x in argv if x])


def override_args(original, changes, drops):
    if not isinstance(original, list) or not all(isinstance(x, str) for x in original):
        raise ValueError('config "args" must be a list of strings')
    pairs = []
    i = 0
    while i < len(original):
        word = original[i]
        if not word.startswith("--"):
            raise ValueError("config args contain a value without a flag")
        flag, sep, value = word.partition("=")
        if not sep and i + 1 < len(original) and not original[i + 1].startswith("--"):
            value, i = original[i + 1], i + 1
            sep = " "
        pairs.append((flag, value if sep else None))
        i += 1
    for flag in drops:
        if not re.fullmatch(r"--[\w-]+", flag):
            raise ValueError("--drop needs a --FLAG")
        pairs = [(f, v) for f, v in pairs if f != flag]
    for change in changes:
        flag, sep, value = change.partition("=")
        if not sep or not re.fullmatch(r"--[\w-]+", flag):
            raise ValueError("--arg needs --FLAG=VALUE")
        pairs = [(f, v) for f, v in pairs if f != flag] + [(flag, value)]
    return [item for flag, value in pairs for item in ([flag] if value is None else [flag, value])]


def corpus():
    found = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    paths = sorted(ROOT / p.decode("utf-8") for p in found.split(b"\0") if p)
    sections = []
    for path in paths:
        if path.suffix.lower() in SUFFIXES and path.is_file():
            sections.append(f"\n### {path.relative_to(ROOT)}\n{path.read_text(encoding='utf-8', errors='replace')}\n")
    if not sections:
        raise ValueError("no tracked source or Markdown files found for the prompt")
    return "".join(sections)


def load_counter(cfg):
    try:
        sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
        import strata_tokenizer as st
        from serve.frontend import ChatTemplate
        folder = Path(cfg["tokenizer"])
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        tok = st.Tokenizer(tokens, (folder / "merges.txt").read_text(encoding="utf-8").split("\n"),
                           json.loads((folder / "token_type.json").read_text(encoding="utf-8")))
        tpl = folder / "chat_template.jinja"
        template = ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja")
        return lambda prompt: len(tok.encode(template.render([{"role": "user", "content": prompt}]),
                                             parse_special=True)), "tokenizer"
    except (ImportError, KeyError, OSError, ValueError) as exc:
        print(f"Tokenizer unavailable ({type(exc).__name__}); using calibrated character estimate.")
        return None, "character estimate"


def build_prompt(target, source, counter, chars_per_token, nonce=None):
    head = f"Benchmark nonce: {nonce or secrets.token_hex(16)}\nFiles:\n"
    tail = "\nNow write a detailed, 600-word technical summary of the files above."
    def text_at(n):
        # Different section numbers keep repeated source text from being identical.
        block = []
        while sum(map(len, block)) < n:
            block.append(f"\n### Section {len(block) + 1}\n{source}")
        return head + "".join(block)[:n] + tail
    count = counter or (lambda s: round(len(s) / chars_per_token))
    low, high = 0, max(1, int(target * chars_per_token * 1.2))
    while count(text_at(high)) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count(text_at(middle)) < target:
            low = middle + 1
        else:
            high = middle
    prompt = text_at(low)
    return prompt, count(prompt)


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong)]


def ram_used_mib():
    if os.name == "nt":
        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError("GlobalMemoryStatusEx failed")
        return round((state.total_phys - state.avail_phys) / 1048576)
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        fields[key] = int(value.strip().split()[0])
    return round((fields["MemTotal"] - fields["MemAvailable"]) / 1024)


def sample_once():
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("nvidia-smi could not read GPU counters")
    gpus = {}
    for line in result.stdout.splitlines():
        uuid, memory, usage = (part.strip() for part in line.split(",", 2))
        gpus[uuid] = {"memory_mib": int(memory), "utilization_pct": int(usage)}
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return {"gpus": gpus, "ram_mib": ram_used_mib()}


def sampler(stop, samples, errors):
    next_sample = time.monotonic()
    while not stop.is_set():
        try:
            samples.append((time.perf_counter(), sample_once()))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            return
        next_sample += 1
        stop.wait(max(0, next_sample - time.monotonic()))


def peaks(samples, start, end):
    active = [item for when, item in samples if start <= when <= end]
    gpus = {}
    for item in active:
        for uuid, values in item["gpus"].items():
            peak = gpus.setdefault(uuid, {"memory_mib": 0, "utilization_pct": 0})
            for key in peak:
                peak[key] = max(peak[key], values[key])
    return {"gpus": gpus, "ram_mib": max((x["ram_mib"] for x in active), default=None)}


def log_metrics(path, offset):
    if not path or not Path(path).exists():
        return {}
    deadline = time.monotonic() + 1
    while True:
        with open(path, "rb") as stream:
            stream.seek(offset)
            lines = stream.read().decode("utf-8", "replace").splitlines()
        values = {}
        for line in lines:
            match = ENGINE_LINE.search(line)
            if match:
                p, pms, pps, g, dms, dps = map(float, match.groups())
                values = {"engine_prompt_tokens": int(p), "engine_prompt_ms": pms,
                          "engine_prefill_tok_s": pps, "engine_generated_tokens": int(g),
                          "engine_decode_ms": dms, "engine_decode_tok_s": dps}
            match = MTP_LINE.search(line)
            if match:
                ok, total = map(int, match.groups())
                values["mtp_acceptance"] = round(ok / total, 3) if total else None
        if values or time.monotonic() >= deadline:
            return values
        time.sleep(0.05)


def request_one(url, content, max_tokens, key, samples, log):
    body = {"model": "strata", "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    offset = Path(log).stat().st_size if log and Path(log).exists() else 0
    start = time.perf_counter()
    first = last = None
    answer = []
    usage = None
    request = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=3600)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"chat request returned HTTP {exc.code}") from None
    with response:
        for raw in response:
            raw = raw.strip()
            if not raw or raw.startswith(b":") or not raw.startswith(b"data: "):
                continue
            if raw == b"data: [DONE]":
                break
            event = json.loads(raw[6:])
            delta = event["choices"][0]["delta"]
            if delta.get("reasoning_content") or delta.get("content") or delta.get("tool_calls"):
                first = first or time.perf_counter()
                last = time.perf_counter()
                answer.append(delta.get("reasoning_content") or delta.get("content") or "")
            usage = event.get("usage") or usage
    end = time.perf_counter()
    if not usage:
        raise RuntimeError("stream ended without a final usage chunk")
    generated = usage["completion_tokens"]
    result = {"prompt_tokens": usage["prompt_tokens"], "generated_tokens": generated,
              "usage": usage,
              "ttft_s": round(first - start, 3) if first else None,
              "prefill_tok_s": round(usage["prompt_tokens"] / (first - start), 2) if first and first > start else None,
              "decode_tok_s": round((generated - 1) / (last - first), 2)
              if first and last and last > first and generated > 1 else None,
              "elapsed_s": round(end - start, 3), "answer_first_200": "".join(answer)[:200], "answer": "".join(answer),
              "peaks": peaks(samples, start, end), "mtp_acceptance": None}
    result.update(log_metrics(log, offset))
    return result


def stop_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def wait_ready(proc, base):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup (code {proc.returncode}); see server.stdout.log and engine log")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                health = json.load(response)
            if health.get("status") == "ok":
                return health
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("server /health did not become ready within 15 minutes")


def log_tail(path, out):
    if not path or not Path(path).exists():
        return
    with open(path, encoding="utf-8", errors="replace") as stream:
        from collections import deque
        lines = deque(stream, maxlen=300)
    (out / "engine-log-tail.txt").write_text("".join(lines), encoding="utf-8")


def write_results(out, data):
    (out / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = ["| context | prompt tokens | TTFT s | prefill tok/s | decode tok/s | MTP acceptance | "
            "peak VRAM GPU0/GPU1 MiB | peak RAM MiB |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for entry in data["contexts"]:
        if entry.get("skipped"):
            rows.append(f"| {entry['target']} | skipped: {entry['skipped']} | | | | | | |")
            continue
        if "median" not in entry:            # the run failed: keep the rows that finished
            rows.append(f"| {entry['target']} | failed | | | | | | |")
            continue
        result = entry["median"]
        gpu = [str(x["memory_mib"]) for x in result["peaks"]["gpus"].values()]
        values = [entry["target"], result["prompt_tokens"], result["ttft_s"], result["prefill_tok_s"],
                  result["decode_tok_s"], result["mtp_acceptance"], "/".join(gpu[:2]) or "—",
                  result["peaks"]["ram_mib"]]
        rows.append("| " + " | ".join("—" if x is None else str(x) for x in values) + " |")
    (out / "results.md").write_text(f"# {data['label'] or 'Strata context benchmark'}\n\n" +
                                     "\n".join(rows) + "\n", encoding="utf-8")


def selftest():
    base = ["--pack", "somewhere", "--spec", "4", "--vision", "--max-context=8192"]
    changed = override_args(base, ["--spec=3", "--pool-workers=7"], ["--vision"])
    assert changed == ["--pack", "somewhere", "--max-context", "8192", "--spec", "3",
                       "--pool-workers", "7"]
    assert base[2:4] == ["--spec", "4"]
    source = "A CUDA engine reads source files and explains its memory layout.\n"
    for target in (1024, 8192):
        prompt, estimate = build_prompt(target, source, lambda text: (len(text) + 3) // 4, 4, "test")
        assert prompt.startswith("Benchmark nonce: test")
        assert abs(estimate - target) / target <= 0.02
        assert len(set(re.findall(r"Section (\d+)", prompt))) > 1
    print("selftest passed: config overrides and prompt lengths within 2%")


def prepare(a):
    if not a.config or not a.out:
        raise ValueError("--config and --out are required")
    if a.max_tokens < 1 or a.repeat < 1:
        raise ValueError("--max-tokens and --repeat must be positive")
    contexts = [int(x.strip()) for x in a.contexts.split(",")]
    if not contexts or any(x < 1 for x in contexts):
        raise ValueError("--contexts needs positive comma-separated lengths")
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    cfg["args"] = override_args(cfg["args"], a.arg, a.drop)
    if a.no_vision:
        cfg.pop("vision", None)
        cfg["args"] = override_args(cfg["args"], [], ["--vision", "--vram-reserve-mib"])
    if a.vision_test and not cfg.get("vision"):
        raise ValueError("--vision-test requires vision in the effective config")
    if a.vision_test and not a.vision_test.is_file():
        raise ValueError("--vision-test image does not exist")
    port = int(cfg["port"])
    if not 1 <= port <= 65535:
        raise ValueError("config port must be 1..65535")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError("project .venv Python is missing")
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise ValueError(f"port {port} is already in use") from None
    sample_once()  # Fail before starting the model if monitoring is unavailable.
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    return cfg, contexts, python, out


def median_run(runs):
    numeric = ("prompt_tokens", "generated_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s",
               "elapsed_s", "mtp_acceptance", "engine_prompt_ms", "engine_prefill_tok_s",
               "engine_decode_ms", "engine_decode_tok_s")
    median = dict(runs[0])
    for field in numeric:
        values = [run[field] for run in runs if run.get(field) is not None]
        if values:
            median[field] = statistics.median(values)
    median["peaks"] = {"gpus": {}, "ram_mib": None}
    for run in runs:
        peak = run["peaks"]
        median["peaks"]["ram_mib"] = max(median["peaks"]["ram_mib"] or 0, peak["ram_mib"] or 0)
        for uuid, counters in peak["gpus"].items():
            old = median["peaks"]["gpus"].setdefault(uuid, {"memory_mib": 0, "utilization_pct": 0})
            for field in old:
                old[field] = max(old[field], counters[field])
    return median


def measure_contexts(a, contexts, data, samples, errors, base, limit, key, source, counter, log):
    ratio = 4.0
    for target in contexts:
        entry = {"target": target, "runs": []}
        data["contexts"].append(entry)
        if target > limit:                       # the context's own size: run it as full as it fits
            if any(e.get("target") == limit for e in data["contexts"]):
                entry["skipped"] = f"over safe limit {limit} (already measured at the limit)"
                print(f"{target}: skipped (safe limit {limit})")
                continue
            print(f"{target}: over the safe limit, measured at {limit} instead")
            entry["clamped_from"], entry["target"] = target, limit
            target = limit
        for rep in range(a.repeat if target <= 8192 else 1):
            nonce = f"{a.nonce}-{target}-{rep}" if a.nonce else None
            prompt, estimate = build_prompt(target, source, counter, ratio, nonce)
            shrink = 0
            while limit < estimate and shrink < 64:  # token boundaries can land a token or two past the target
                shrink += estimate - limit
                prompt, estimate = build_prompt(target - shrink, source, counter, ratio, nonce)
            if estimate > limit:
                raise ValueError(f"prompt estimate {estimate} exceeds safe limit {limit}")
            result = request_one(base + "/v1/chat/completions", prompt, a.max_tokens, key, samples, log)
            result["estimated_prompt_tokens"] = estimate
            entry["runs"].append(result)
            if not counter:
                ratio *= estimate / result["prompt_tokens"]
            if errors:
                raise RuntimeError("hardware sampling failed: " + errors[0])
        entry["median"] = median_run(entry["runs"])
        print(f"{target}: {entry['median']['prompt_tokens']} prompt tokens, TTFT {entry['median']['ttft_s']} s")


def measure_vision(a, data, base, key, samples, log):
    mime = mimetypes.guess_type(a.vision_test.name)[0] or "image/jpeg"
    encoded = base64.b64encode(a.vision_test.read_bytes()).decode("ascii")
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
               {"type": "text", "text": "Describe this image in detail."}]
    data["vision"] = request_one(base + "/v1/chat/completions", content, a.max_tokens, key, samples, log)


def main():
    a = cli()
    if a.selftest:
        selftest()
        return
    cfg, contexts, python, out = prepare(a)
    counter, method = load_counter(cfg)
    source = corpus()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", prefix=".ctx-bench-",
                                     dir=out, delete=False) as config_file:
        json.dump(cfg, config_file, indent=2)
        tmp = Path(config_file.name)
    base = f"http://127.0.0.1:{cfg['port']}"
    data = {"label": a.label, "config": str(a.config), "engine_args": cfg["args"],
            "token_count_method": method, "max_tokens": a.max_tokens, "contexts": [], "vision": None}
    stop, samples, errors = threading.Event(), [], []
    proc = None
    log = cfg.get("log")
    thread = threading.Thread(target=sampler, args=(stop, samples, errors), daemon=True)
    try:
        with open(out / "server.stdout.log", "w", encoding="utf-8") as stdout:
            proc = subprocess.Popen([str(python), str(ROOT / "serve/server.py"), "--engine", "strata",
                                     "--config", str(tmp), "--port", str(cfg["port"])], cwd=ROOT, stdout=stdout,
                                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                    start_new_session=os.name != "nt")
            health = wait_ready(proc, base)
            data["health"] = health
            limit = health["max_context"] - a.max_tokens - 512
            key = os.environ.get("STRATA_API_KEY") or cfg.get("api_key", "")
            thread.start()
            measure_contexts(a, contexts, data, samples, errors, base, limit, key, source, counter, log)
            if a.vision_test:
                measure_vision(a, data, base, key, samples, log)
    finally:
        stop.set()
        if proc:
            stop_tree(proc)
        if thread.ident:
            thread.join(timeout=11)
        try:
            log_tail(log, out)
            write_results(out, data)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"Results: {out / 'results.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; server process tree stopped.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
