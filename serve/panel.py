"""Strata control panel: start / stop / switch the installed models (the configs/strata-*.json setup wrote),
the API's address and key, GPU / RAM use, and what the model is doing.

    python serve/panel.py [--port 8091] [--open]      PANEL.bat starts it without a window
    python serve/panel.py --start iq3_xxs --open       START-HERE.bat: turn a model on (the panel starts if needed)

The panel only starts and stops serve/server.py (one model at a time); the API itself is served by server.py on the
model's port, so apps keep working when the panel is closed.  Stopping a model ends its process tree, which frees
all of its VRAM and RAM.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = Path(__file__).with_name("panel.html")
CONFIGS = ROOT / "configs"                          # strata-<model>.json, written by setup.py
STATE_DIR = ROOT / ".panel"                         # the API key and the running model (gitignored)
APP_NAME = "strata-panel"
PANEL_PORT = 8091
BOOT_TIMEOUT_S = 900.0                              # a cold start reads ~45 GB into RAM
MAX_BODY = 64 * 1024
WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if WIN else 0                # CREATE_NO_WINDOW
# one line per request in the engine log (engine 0.1.3+)
REQUEST_LINE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")


class PanelError(Exception):
    pass


EXTERNAL = "這個模型是在面板外開的，面板找不到它的程序：關掉它的視窗後再從這裡開"


# ------------------------------------------------------------------------------------------------ models
def arg(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def load_models() -> dict[str, dict[str, Any]]:
    models = {}
    for path in sorted(CONFIGS.glob("strata-*.json")):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("exe"), str) and isinstance(cfg.get("args"), list):
            models[path.stem.removeprefix("strata-")] = {"path": path, "cfg": cfg}
    return models


def ctx_label(ctx: int) -> str:
    return "262K" if ctx == 262144 else f"{ctx // 1024}K"      # 2^18 is called 262K everywhere else


def expert_uuid_of(cfg: dict) -> str | None:
    """The second card's uuid, or None when the config has no expert cache or the value cannot be one uuid.
    The same card as the main GPU is rejected: ordinal 1 has to be a different device."""
    if not isinstance(cfg, dict):
        return None
    raw = cfg.get("expert_gpu")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    main = cfg.get("gpu")
    main_s = main.strip() if isinstance(main, str) else ""
    if not text or text == main_s or "," in text or any(c.isspace() for c in text):
        return None
    return text


def expert_fact(cfg: dict) -> str | None:
    """Model-card text. None hides the row, so a config without expert_gpu looks as it does today.
    A present but unusable value is shown, not dropped."""
    if not isinstance(cfg, dict):
        return None
    if "expert_gpu" in cfg and cfg.get("expert_gpu") is not None and expert_uuid_of(cfg) is None:
        return "設定無效"
    uid = expert_uuid_of(cfg)
    if uid is None:
        return None
    vision = cfg.get("vision")
    if isinstance(vision, dict) and vision.get("gpu_uuid") == uid:
        return "看圖＋專家"
    return "第二張卡"


# one card can read images and hold the expert cache; setdefault would keep the first label and hide the second
_GPU_ROLES = ("跑模型", "看圖", "專家")


def assign_gpu_roles(infos: list[dict[str, Any]]) -> dict[str, str]:
    """Role per GPU uuid. The second expert cache is 專家, or 看圖＋專家 when the image encoder shares that card."""
    tags: dict[str, set[str]] = {}

    def add(uuid: Any, role: str) -> None:
        if isinstance(uuid, str) and uuid:
            tags.setdefault(uuid, set()).add(role)

    for info in infos:
        if not info:
            continue
        add(info.get("gpu_uuid"), "跑模型")
        add(info.get("vision_uuid"), "看圖")
        add(info.get("expert_uuid"), "專家")
    return {uuid: "＋".join(role for role in _GPU_ROLES if role in got) for uuid, got in tags.items()}


def model_info(model_id: str, path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    args = [str(a) for a in cfg["args"]]
    name = cfg.get("model_name", model_id)
    ctx = int(arg(args, "--max-context") or 0) or None
    vision = cfg.get("vision") or {}
    exe = Path(cfg["exe"])
    quant = next((p for p in model_id.split("-") if re.fullmatch(r"i?q\d\w*", p)), None)
    return {
        "id": model_id,
        "label": ("去審查版" if "uncensored" in name else "原版") + f" · {quant.upper() if quant else model_id}"
                 + (f" · {ctx_label(ctx)}" if ctx else ""),
        "ctx_label": ctx_label(ctx) if ctx else None,
        "model_id": name,
        "config": path.name,
        "n_ctx": ctx,
        "kv": arg(args, "--kv") or "fp16",
        "mtp": f"猜 {arg(args, '--spec')} 個字" if "--mtp" in args and arg(args, "--spec") else "關",
        "vision": ("另一張卡" if vision.get("gpu_uuid") else "GPU" if vision.get("gpu") else "CPU") if vision else "關",
        "prefill": arg(args, "--prefill"),
        "pcie_frac": arg(args, "--pcie-frac"),
        "engine": f"{exe.parent.name}\\{exe.name}",
        "port": int(cfg.get("port") or 8080),
        "gpu_uuid": cfg.get("gpu"),
        "vision_uuid": vision.get("gpu_uuid"),
        "expert_uuid": expert_uuid_of(cfg),
        "expert": expert_fact(cfg),
        "ready": exe.is_file() and Path(arg(args, "--pack") or exe).exists(),
        "engine_log": cfg.get("log"),
    }


# ------------------------------------------------------------------------------------------------ supervisor
@dataclass
class State:
    status: str = "off"                             # off / starting / ready / stopping / error / external
    model: str | None = None
    pid: int | None = None
    port: int | None = None
    log_offset: int = 0                             # engine log size when this run started: stats count from here
    started_at: float | None = None
    ready_at: float | None = None
    error: str | None = None
    generation: int = 0
    proc: subprocess.Popen | None = None

    def saved(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("model", "pid", "port", "log_offset", "started_at", "ready_at")}


class Supervisor:
    def __init__(self) -> None:
        STATE_DIR.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.state = State()
        self._adopt()

    # ---- API key: required on /v1/*; $STRATA_API_KEY wins over the saved one
    @property
    def key_file(self) -> Path:
        return STATE_DIR / "api-key.txt"

    def api_key(self) -> str:
        env = os.environ.get("STRATA_API_KEY", "").strip()
        if env:
            return env
        saved = self.key_file.read_text(encoding="utf-8").strip() if self.key_file.is_file() else ""
        return saved or self._new_key()

    def _new_key(self) -> str:
        key = "sk-strata-" + secrets.token_urlsafe(32)
        self.key_file.write_text(key + "\n", encoding="utf-8")
        return key

    def rotate_key(self) -> str:
        if os.environ.get("STRATA_API_KEY"):
            raise PanelError("API key 由環境變數 STRATA_API_KEY 指定，不能在面板重產")
        with self.lock:
            key = self._new_key()
            running = self.state.model if self.state.status in {"starting", "ready"} else None
        if running:
            self.start(running)                     # server.py reads the key when it starts
        return key

    # ---- lifecycle
    def start(self, model_id: str) -> None:
        try:
            self._start(model_id)
        except PanelError as exc:
            with self.lock:                          # pythonw has no console: the page shows why
                if self.state.status in {"off", "error"}:
                    self.state.status, self.state.error = "error", str(exc)
            raise

    def _start(self, model_id: str) -> None:
        models = load_models()
        if model_id not in models:
            raise PanelError(f"找不到模型設定：configs/strata-{model_id}.json")
        info = model_info(model_id, models[model_id]["path"], models[model_id]["cfg"])
        if not info["ready"]:
            raise PanelError(f"{info['label']}：引擎或模型檔不存在，先跑 START-HERE.bat --setup")
        with self.lock:
            self._refresh_locked(models)            # a model started outside the panel is taken over first
            if self.state.status == "external":
                raise PanelError(EXTERNAL)
            self.state.generation += 1
            gen = self.state.generation
            self._kill_locked()
            log = Path(info["engine_log"]) if info["engine_log"] else None
            self.state = State(status="starting", model=model_id, port=info["port"], started_at=time.time(),
                               log_offset=log.stat().st_size if log and log.is_file() else 0, generation=gen)
        threading.Thread(target=self._boot, args=(models[model_id]["path"], info["port"], gen), daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self._refresh_locked(load_models())
            if self.state.status == "external":
                raise PanelError(EXTERNAL)
            self.state.generation += 1
            self.state.status = "stopping"
            self._kill_locked()
            self.state = State(generation=self.state.generation)
            self._state_file().unlink(missing_ok=True)

    def _boot(self, cfg_path: Path, port: int, gen: int) -> None:
        try:
            deadline = time.monotonic() + 15            # the model just stopped may still hold the port a moment
            while port_open(port) and time.monotonic() < deadline:
                time.sleep(0.5)
            if port_open(port):
                raise PanelError(f"127.0.0.1:{port} 已被其他程式佔用")
            proc = self._spawn(cfg_path, port)
        except (PanelError, OSError) as exc:
            self._fail(gen, str(exc))
            return
        with self.lock:
            if gen != self.state.generation:
                kill_tree(proc.pid)
                return
            self.state.proc, self.state.pid = proc, proc.pid
            self._state_file().write_text(json.dumps(self.state.saved()), encoding="utf-8")
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if gen != self.state.generation:
                return
            if proc.poll() is not None:
                self._fail(gen, f"server.py 結束了（exit={proc.returncode}），看下方 log")
                return
            if health(port) is not None:
                with self.lock:
                    if gen == self.state.generation:
                        self.state.status, self.state.ready_at = "ready", time.time()
                        self._state_file().write_text(json.dumps(self.state.saved()), encoding="utf-8")
                return
            time.sleep(1)
        self._fail(gen, f"{BOOT_TIMEOUT_S / 60:.0f} 分鐘內沒有就緒")

    def _spawn(self, cfg_path: Path, port: int) -> subprocess.Popen:
        env = {**os.environ, "STRATA_API_KEY": self.api_key(), "PYTHONUNBUFFERED": "1"}   # not on the command line
        out = self.server_log.open("w", encoding="utf-8")
        try:
            return subprocess.Popen(
                [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
                 "--port", str(port)],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                creationflags=NO_WINDOW, start_new_session=not WIN)
        finally:
            out.close()

    def _fail(self, gen: int, message: str) -> None:
        with self.lock:
            if gen != self.state.generation:
                return
            self._kill_locked()
            self.state.status, self.state.error = "error", message
            self._state_file().unlink(missing_ok=True)

    def _kill_locked(self) -> None:
        if self.state.pid is not None and (self.state.proc is not None or is_python(self.state.pid)):
            kill_tree(self.state.pid)
        self.state.proc, self.state.pid = None, None

    @property
    def server_log(self) -> Path:
        return STATE_DIR / "server.log"

    def _state_file(self) -> Path:
        return STATE_DIR / "state.json"

    def _adopt(self) -> None:
        """A reopened panel takes over the model it started last time, if that is still running."""
        try:
            saved = json.loads(self._state_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        pid, port = saved.get("pid"), saved.get("port")
        if not (isinstance(pid, int) and isinstance(port, int) and saved.get("model") in load_models()
                and is_python(pid)):
            self._state_file().unlink(missing_ok=True)
            return
        self.state = State(status="ready" if health(port) is not None else "starting", model=saved["model"], pid=pid,
                           port=port, log_offset=int(saved.get("log_offset") or 0), started_at=saved.get("started_at"),
                           ready_at=saved.get("ready_at") or time.time())

    # ---- status
    def _refresh_locked(self, models: dict[str, dict[str, Any]]) -> None:
        s = self.state
        if s.status in {"starting", "ready"} and s.pid is not None:
            alive = s.proc.poll() is None if s.proc is not None else is_python(s.pid)
            if not alive:
                s.status, s.error, s.proc, s.pid = "error", "server.py 意外結束，看下方 log", None, None
                self._state_file().unlink(missing_ok=True)
            elif s.status == "ready" and engine_stopped(s.port):   # server.py up, its engine gone
                self._kill_locked()
                s.status, s.error = "error", "引擎意外結束（看引擎 log）；再按一次開關就會重開"
                self._state_file().unlink(missing_ok=True)
            elif s.status == "starting" and s.proc is None and health(s.port) is not None:
                s.status = "ready"
        elif s.status in {"off", "external", "error"}:
            self._take_over_locked(models)

    def _take_over_locked(self, models: dict[str, dict[str, Any]]) -> None:
        """A model started outside the panel (START-HERE.bat, an old run script, by hand) is taken over: from then on
        the panel stops and switches it like its own.  Only when its process cannot be found is it shown read-only."""
        s = self.state
        ports = {model_info(m, v["path"], v["cfg"])["port"] for m, v in models.items()} or {8080}
        found = next(((p, h) for p in ports if (h := health(p)) is not None), None)
        if not found:
            if s.status == "external":
                self.state = State(generation=s.generation)
            return
        port, h = found
        model = next((m for m, v in models.items() if v["cfg"].get("model_name") == h.get("model")), None)
        pid = listener_pid(port)
        if pid is None or not is_python(pid):
            s.status, s.port, s.model = "external", port, model
            return
        log = models[model]["cfg"].get("log") if model else None
        self.state = State(status="ready", model=model, pid=pid, port=port, ready_at=time.time(),
                           log_offset=Path(log).stat().st_size if log and Path(log).is_file() else 0,
                           generation=s.generation)
        self._state_file().write_text(json.dumps(self.state.saved()), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        models = load_models()
        infos = [model_info(m, v["path"], v["cfg"]) for m, v in models.items()]
        with self.lock:
            self._refresh_locked(models)
            s = self.state
            snap = {"status": s.status, "model": s.model, "pid": s.pid, "started_at": s.started_at,
                    "ready_at": s.ready_at, "error": s.error}
            port, offset = s.port or (infos[0]["port"] if infos else 8080), s.log_offset
        active = next((i for i in infos if i["id"] == snap["model"]), None)
        h = health(port) if snap["status"] == "ready" else None
        snap["warning"] = ("看圖程式已結束，傳圖片會失敗；關掉再開這個模型就會恢復"
                           if h and active and active["vision"] != "關" and not h.get("images") else None)
        engine_log = Path(active["engine_log"]) if active and active["engine_log"] else None
        base = f"http://127.0.0.1:{port}"
        return {
            "server": snap,
            "api": {"base_url": base, "openai_base_url": f"{base}/v1", "anthropic_base_url": base,
                    "chat_url": f"{base}/", "api_key": self.api_key(),
                    "key_required": bool(h.get("api_key")) if h else None,
                    "key_from_env": bool(os.environ.get("STRATA_API_KEY"))},
            "models": infos,
            "gpus": query_gpus(infos, active),
            "gpu_apps": query_gpu_apps(gpu_roles(infos, active)),
            "ram": ram_usage(),
            "live": live_status(port) if snap["status"] in {"ready", "external"} else None,
            "session": session_stats(engine_log, offset if snap["status"] != "external" else 0),
            "server_log": tail(self.server_log) if snap["status"] != "external" else "",
            "engine_log": tail(engine_log) if engine_log else "",
            "now": time.time(),
        }


# ------------------------------------------------------------------------------------------------ probes
def health(port: int | None) -> dict[str, Any] | None:
    if not port:
        return None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            data = json.loads(r.read())
            return data if isinstance(data, dict) and data.get("status") == "ok" else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def engine_stopped(port: int | None) -> bool:
    """server.py answers /health with 503 "engine stopped" once its engine process has ended."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).close()
    except urllib.error.HTTPError as exc:
        return exc.code == 503
    except (urllib.error.URLError, OSError):
        pass
    return False


def live_status(port: int) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=1) as r:
            data = json.loads(r.read())
            return data if isinstance(data, dict) else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def listener_pid(port: int) -> int | None:
    """The process listening on 127.0.0.1:port (server.py), from netstat (Windows) or ss (Linux)."""
    try:
        if WIN:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, errors="replace",
                                 timeout=5, creationflags=NO_WINDOW).stdout
            for f in (line.split() for line in out.splitlines()):
                # the remote address of a listening socket is 0.0.0.0:0 (the state column is translated)
                if len(f) == 5 and f[1].rsplit(":", 1)[-1] == str(port) and f[2] in {"0.0.0.0:0", "[::]:0"}:
                    return int(f[4])
            return None
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def session_stats(log: Path | None, offset: int) -> dict[str, Any]:
    """Per-request lines the engine wrote since this model started: count, last and average speeds, MTP hits."""
    rows = []
    if log and log.is_file():
        try:
            with log.open("rb") as f:
                size = f.seek(0, os.SEEK_END)
                f.seek(offset if offset <= size else 0)
                if size - f.tell() > 4 * 2**20:     # a long session: its last 4 MB is enough
                    f.seek(size - 4 * 2**20)
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            text = ""
        rows = [m.groups() for m in REQUEST_LINE.finditer(text)]
    if not rows:
        return {"requests": 0}
    num = [[float(x) if x is not None else 0.0 for x in r] for r in rows]
    read, read_ms = sum(r[2] for r in num), sum(r[3] for r in num)
    gen, gen_ms = sum(r[5] for r in num), sum(r[6] for r in num)
    ok, drafts = sum(r[8] for r in num), sum(r[9] for r in num)
    last = num[-1]
    return {"requests": len(rows), "last_prompt": int(last[0]), "last_reused": int(last[1]),
            "last_prefill_tps": last[4], "last_decode_tps": last[7], "last_generated": int(last[5]),
            "avg_prefill_tps": read / read_ms * 1000 if read_ms else None,
            "avg_decode_tps": gen / gen_ms * 1000 if gen_ms else None,
            "prompt_tokens": int(sum(r[0] for r in num)), "read_tokens": int(read), "generated_tokens": int(gen),
            "mtp_accept": ok / drafts if drafts else None}


def query_gpus(infos: list[dict[str, Any]], active: dict[str, Any] | None) -> list[dict[str, Any]]:
    keys = ("uuid", "index", "name", "memory_used_mib", "memory_total_mib", "util_pct", "temp_c", "power_w",
            "power_limit_w")
    roles = gpu_roles(infos, active)
    gpus = []
    for row in smi(["--query-gpu=uuid,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
                    "power.draw,power.limit", "--format=csv,noheader,nounits"]):
        if len(row) != len(keys):
            continue
        g: dict[str, Any] = dict(zip(keys, row))
        for k in keys[3:]:
            g[k] = number(g[k])
        g["role"] = roles.get(g["uuid"])
        gpus.append(g)
    return gpus


def gpu_roles(infos: list[dict[str, Any]], active: dict[str, Any] | None) -> dict[str, str]:
    # the model that is on; with nothing running, every installed config, as before
    return assign_gpu_roles([active] if active else infos)


def query_gpu_apps(roles: dict[str, str]) -> list[dict[str, Any]]:
    """Strata's own processes on the GPUs (WDDM lists every desktop app, without per-app memory).
    The engine runs the second expert card from a helper strata.exe that sees only that card (docs/DUAL-GPU.md)."""
    apps: dict[str, dict[str, Any]] = {}
    for row in smi(["--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader,nounits"]):
        if len(row) == 4 and "strata" in row[1].lower():
            entry = apps.setdefault(row[0], {"pid": row[0], "name": Path(row[1].replace("\\", "/")).name,
                                             "memory_mib": number(row[2]), "gpus": 0, "on": []})
            entry["gpus"] += 1
            entry["on"].append(roles.get(row[3]) or "")
    for a in apps.values():
        on = a.pop("on")
        a["role"] = ("第二張卡的專家助手" if a["name"].lower() == "strata.exe"
                     and all("專家" in r and "跑模型" not in r for r in on) else None)
    return list(apps.values())


def ram_usage() -> dict[str, float] | None:
    if WIN:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return {"total_gib": m.total / 2**30, "used_gib": (m.total - m.avail) / 2**30}
    try:
        info = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if len(l.split()) >= 2}
    except OSError:
        return None
    return {"total_gib": info["MemTotal"] / 2**20, "used_gib": (info["MemTotal"] - info["MemAvailable"]) / 2**20}


def smi(args: list[str]) -> list[list[str]]:
    try:
        done = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=5, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [[p.strip() for p in line.split(",")] for line in done.stdout.splitlines() if line.strip()]


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def tail(path: Path | None, max_bytes: int = 16000, lines: int = 80) -> str:
    if not path:
        return ""
    try:
        with path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def is_python(pid: int) -> bool:
    """The saved pid is still a Python process (server.py), not a reused pid of something else."""
    if WIN:
        done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                              text=True, errors="replace", creationflags=NO_WINDOW)
        return "python" in done.stdout.lower()
    try:
        return "python" in Path(f"/proc/{pid}/comm").read_text(encoding="utf-8")
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    """server.py plus its engine and image encoder: ending the whole tree frees their VRAM and RAM."""
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and is_python(pid):
            time.sleep(0.3)
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(sup: Supervisor, token: str, port: int, shutdown: threading.Event):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    html = HTML_PATH.read_text(encoding="utf-8").replace("__PANEL_TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if not self._host_ok():
                return
            if self.path == "/api/ping":
                self._json(200, {"success": True, "data": {"app": APP_NAME}, "error": None})
            elif self.path == "/api/status":
                self._json(200, {"success": True, "data": sup.status(), "error": None})
            elif self.path in {"/", "/index.html"}:
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._json(404, {"success": False, "data": None, "error": "not found"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return
            # a custom header forces a CORS preflight (never answered), so other pages cannot switch models
            if not secrets.compare_digest(self.headers.get("X-Panel-Token", ""), token):
                self._json(403, {"success": False, "data": None, "error": "bad panel token"})
                return
            try:
                data = self._dispatch(self._body())
            except (PanelError, ValueError) as exc:
                self._json(400, {"success": False, "data": None, "error": str(exc)})
                return
            self._json(200, {"success": True, "data": data, "error": None})

        def _dispatch(self, body: dict[str, Any]) -> Any:
            if self.path == "/api/start":
                if not isinstance(body.get("model"), str):
                    raise ValueError("model 必須是字串")
                sup.start(body["model"])
            elif self.path == "/api/stop":
                sup.stop()
            elif self.path == "/api/rotate-key":
                return {"api_key": sup.rotate_key()}
            elif self.path == "/api/quit":
                if sup.state.status != "external":
                    sup.stop()
                shutdown.set()
            else:
                raise ValueError(f"未知操作：{self.path}")
            return None

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body 太大")
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(payload, dict):
                raise ValueError("body 必須是 JSON 物件")
            return payload

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") in hosts:   # blocks DNS rebinding
                return True
            self._json(421, {"success": False, "data": None, "error": "bad host"})
            return False

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def _send(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


class PanelServer(ThreadingHTTPServer):
    allow_reuse_address = not WIN                   # on Windows SO_REUSEADDR lets a second server take a used port
    daemon_threads = True


def panel_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1) as r:
            return json.loads(r.read()).get("data", {}).get("app") == APP_NAME
    except (urllib.error.URLError, OSError, ValueError):
        return False


def request_start(port: int, model: str) -> None:
    """Ask the running panel to turn a model on (the page carries the token its buttons use)."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
        m = re.search(r'const TOKEN = "([^"]+)"', r.read().decode("utf-8", errors="replace"))
    if not m:
        raise PanelError("the panel page has no token")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/start", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json", "X-Panel-Token": m.group(1)})
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except urllib.error.HTTPError as exc:
        raise PanelError(json.loads(exc.read() or b"{}").get("error") or f"HTTP {exc.code}") from None


def selftest() -> int:
    log = STATE_DIR / "selftest.log"
    STATE_DIR.mkdir(exist_ok=True)
    log.write_text("old line\nstrata serve: prompt 1024 tokens = 0 reused + 1024 read in 5000 ms (204.8 tok/s), "
                   "400 generated in 10000 ms (40.0 tok/s), drafts accepted 30 of 40, 1 checkpoints\n"
                   "strata serve: prompt 2048 tokens = 1024 reused + 1024 read in 2500 ms (409.6 tok/s), "
                   "100 generated in 5000 ms (20.0 tok/s)\n", encoding="utf-8")
    s = session_stats(log, 0)
    log.unlink()
    assert s["requests"] == 2 and s["last_prompt"] == 2048 and s["last_reused"] == 1024, s
    assert abs(s["avg_prefill_tps"] - 2048 / 7.5) < 1e-6 and abs(s["avg_decode_tps"] - 500 / 15) < 1e-6, s
    assert s["mtp_accept"] == 0.75, s
    assert session_stats(None, 0) == {"requests": 0}
    assert arg(["--kv", "int8", "--spec"], "--kv") == "int8" and arg(["--spec"], "--spec") is None
    for m, v in load_models().items():
        info = model_info(m, v["path"], v["cfg"])
        print(f"  {info['label']:28s} {info['config']:34s} ready={info['ready']}")
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PANEL_PORT)
    ap.add_argument("--open", action="store_true", help="open the panel in the browser")
    ap.add_argument("--start", metavar="MODEL", help="turn this model on (strata-MODEL.json), starting the panel if needed")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    url = f"http://127.0.0.1:{a.port}/"
    if panel_running(a.port):                        # already open: hand it the model, show it
        code = 0
        if a.start:
            try:
                request_start(a.port, a.start)
            except (PanelError, OSError, ValueError) as exc:
                print(f"panel: {exc}", file=sys.stderr)
                code = 1
        if a.open:
            webbrowser.open(url)
        return code
    sup = Supervisor()
    shutdown = threading.Event()
    try:
        httpd = PanelServer(("127.0.0.1", a.port), make_handler(sup, secrets.token_urlsafe(24), a.port, shutdown))
    except OSError:
        print(f"panel: 127.0.0.1:{a.port} is taken by another program; use --port", file=sys.stderr)
        return 1
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"panel: {url}", flush=True)
    if a.start:
        try:
            sup.start(a.start)
        except PanelError:                           # kept in the state: the page shows it
            pass
    if a.open:
        webbrowser.open(url)
    try:
        shutdown.wait()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
