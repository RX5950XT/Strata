"""Real-model dual-GPU regression: python bench/expert_gpu_smoke.py [single|dual|missing]."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "build-dual" / "expert-gpu-smoke"
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer



def run(mode):
    config = json.loads((ROOT / "configs/strata-iq3_xxs.json").read_text(encoding="utf-8"))
    args = config["args"][:]
    for flag in [a for a in args if a.startswith("--expert-gpu")]:   # the installed config may already be dual
        i = args.index(flag)
        del args[i:i + 2]
    args.remove("--vision")
    for flag, value in (("--max-context", "8192"), ("--prefill", "2048")):
        args[args.index(flag) + 1] = value
    directory = ROOT / "packs/iq3_xxs/tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").split("\n"),
                          json.loads((directory / "token_type.json").read_text()))
    prompt = "<|im_start|>user\nExplain how a rainbow forms, step by step, then describe a simple experiment a child can do to see one. Give a detailed answer in English.<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, parse_special=True)
    args += ["--tokens", ",".join(map(str, ids)), "--temperature", "0", "--max-new", "200"]
    if mode != "single":
        args += ["--expert-gpu", "1" if mode == "dual" else "5"]
    args += os.environ.get("SMOKE_EXTRA", "").split()   # e.g. SMOKE_EXTRA="--expert-gpu-pcie-frac 0.4"
    command = [str(ROOT / "build-dual/strata.exe"), *args]
    env = os.environ.copy()
    # the main card first, then the second one: CUDA orders the fastest card first unless this is set
    second = config.get("expert_gpu") or (config.get("vision") or {}).get("gpu_uuid")
    env["CUDA_VISIBLE_DEVICES"] = f"{config['gpu']},{second}"
    env["PATH"] = os.pathsep.join(config.get("lib_dirs", [])) + os.pathsep + env["PATH"]
    LOGS.mkdir(exist_ok=True)
    (LOGS / f"{mode}.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    memory = []
    start = time.monotonic()
    with (LOGS / f"{mode}.stdout").open("w") as stdout, (LOGS / f"{mode}.stderr").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        print(f"{mode}: PID {process.pid}", flush=True)
        try:
            while process.poll() is None:
                if time.monotonic() - start > 900:
                    raise TimeoutError(f"{mode}: engine exceeded 900 seconds")
                snapshot = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=uuid,memory.used,memory.free", "--format=csv,noheader,nounits"
                ], text=True)
                memory.append(snapshot.strip())
                time.sleep(3)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    output = (LOGS / f"{mode}.stdout").read_text(errors="replace")
    errors = (LOGS / f"{mode}.stderr").read_text(errors="replace")
    (LOGS / f"{mode}.memory.json").write_text(json.dumps(memory), encoding="utf-8")
    assert process.returncode == 0, errors[-4000:]
    generated = re.search(r"^output\s*:\s*([\d ]+)$", output, re.MULTILINE)
    assert generated, output[-4000:]
    produced = list(map(int, generated[1].split()))
    assert len(produced) == 200, len(produced)
    text = tokenizer.decode(produced)
    (LOGS / f"{mode}.tokens.json").write_text(json.dumps(produced), encoding="utf-8")
    (LOGS / f"{mode}.text.txt").write_text(text, encoding="utf-8")
    if mode == "dual":
        assert "expert GPU: " in errors and "disabled:" not in errors, errors[-4000:]
        assert re.search(r"expert GPU: [1-9]\d* experts", errors), errors[-4000:]
    if mode == "missing":
        assert errors.count("expert GPU 5 disabled:") == 1, errors[-4000:]
    for line in output.splitlines():
        if line.startswith(("decode ", "verify window", "pool multi")):
            print(f"{mode}: {line}", flush=True)
    for line in errors.splitlines():
        if "expert GPU" in line or line.startswith("strata generate: R4 hit"):
            print(f"{mode}: {line}", flush=True)
    print(text[:500], flush=True)


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["single", "dual", "missing"]:
        assert mode in ("single", "dual", "missing")
        run(mode)
    paths = [LOGS / f"{mode}.tokens.json" for mode in ("single", "dual")]
    if all(path.exists() for path in paths):
        first, second = [json.loads(path.read_text()) for path in paths]
        prefix = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), len(first))
        print(f"single/dual matching prefix: {prefix}/{len(first)} tokens", flush=True)
        assert prefix >= 8, f"early divergence at token {prefix}"
