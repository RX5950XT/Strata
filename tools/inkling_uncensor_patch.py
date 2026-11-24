#!/usr/bin/env python3
"""fork: make an uncensored Inkling-2-Mini .strata from an official Strata build (SQ-10 / SQ-15 / SQ-20).

OpenUncensored's Inkling-2-Mini-Uncensored (the source of Abiray's GGUFs) changes only the 20 attention output
projections `model.llm.layers.{4..23}.attn.wo_ud.weight` (BF16); experts, vision and DSpark are the original bytes.
The official builds store each wo_ud as Q8_ROW ([rows] f32 scale = absmax/127, then [rows, cols] int8 =
round-half-even(w / scale)) or as BF16 (SQ-15's global layers 5/11/17/23, all of SQ-20; copied as is). The encoder
reproduces the official bytes exactly on the unedited layers 0..3 (checked below). The tool copies the build, writes
the uncensored wo_ud into the same slots (same sizes, Section 1 only), and rewrites Section 1's SHA-256 in the header
so `--selfcheck --verify-hash` still passes.

Usage: python -I tools/inkling_uncensor_patch.py SRC.strata OUT.strata
The uncensored BF16 tensors are read with HTTP range requests (about 340 MB) from the pinned Hugging Face commit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import zstandard

REPO = "OpenUncensored/Inkling-2-Mini-Uncensored"
REV = "4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487"   # 2026-11-21; weights unchanged since 454b128 (later commits: model card)
FILE = "model-00001-of-00001.safetensors"
EDITED = range(4, 24)          # layers OpenUncensored edited (README: wo_ud on layers 4 to 23)
CHECK = range(0, 4)            # unedited layers: our Q8_ROW bytes must equal the official ones
ROWS, COLS = 2304, 3072
SLOT_BYTES = {"Q8_ROW": ROWS * 4 + ROWS * COLS, "BF16": ROWS * COLS * 2}   # the two wo_ud formats SQ-10/15/20 use


def http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "strata-fork-patch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"short read {len(data)} != {end - start + 1}")
    return data


def read_header(path: Path) -> tuple[bytes, list[tuple]]:
    with path.open("rb") as f:
        frame = f.read(4096)
    if frame[:8] != b"STRATA\x00\x01":
        raise ValueError(f"{path} is not a .strata file")
    entries = [struct.unpack("<QQ32sIIQ", frame[16 + 64 * i:16 + 64 * (i + 1)]) for i in range(6)]
    return frame, entries


def manifest(path: Path, entries) -> dict:
    sec0 = next(e for e in entries if e[3] == 0)
    with path.open("rb") as f:
        f.seek(sec0[0])
        raw = f.read(sec0[1])
    return json.loads(zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 28))


def q8_row(w: np.ndarray) -> bytes:
    scale = (np.abs(w).max(axis=1) / 127).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / safe[:, None]), -127, 127).astype(np.int8)
    return scale.astype("<f4").tobytes() + q.tobytes()


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(ROWS, COLS)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(argv[1]), Path(argv[2])
    frame, entries = read_header(src)
    man = manifest(src, entries)
    tensors = {t["name"]: t for t in man["trunk_tensors"]}
    sec1 = next(e for e in entries if e[3] == 1)

    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{FILE}"
    n = struct.unpack("<Q", http_range(url, 0, 7))[0]
    st = json.loads(http_range(url, 8, 8 + n - 1))
    base = 8 + n

    def wanted(layer: int) -> tuple[dict, bytes]:
        name = f"model.llm.layers.{layer}.attn.wo_ud.weight"
        t = tensors[name]
        if t["shape"] != [ROWS, COLS] or t["bytes"] != SLOT_BYTES.get(t["format"]):
            raise ValueError(f"{name}: unexpected {t['format']} {t['shape']} {t['bytes']}")
        info = st[name]
        if info["dtype"] != "BF16" or info["shape"] != [ROWS, COLS]:
            raise ValueError(f"{name}: source is {info['dtype']} {info['shape']}")
        a, b = info["data_offsets"]
        raw = http_range(url, base + a, base + b - 1)
        return t, raw if t["format"] == "BF16" else q8_row(bf16_to_f32(raw))

    with src.open("rb") as f:
        for layer in CHECK:
            t, enc = wanted(layer)
            f.seek(sec1[0] + t["offset"])
            if f.read(t["bytes"]) != enc:
                raise SystemExit(f"layer {layer}: our {t['format']} bytes differ from the official build; not patching")
    print(f"layers {CHECK.start}..{CHECK.stop - 1}: wo_ud bytes match the official build exactly")

    tmp = out.with_suffix(out.suffix + ".part")
    shutil.copyfile(src, tmp)
    with tmp.open("r+b") as f:
        for layer in EDITED:
            t, enc = wanted(layer)
            f.seek(sec1[0] + t["offset"])
            f.write(enc)
        print(f"patched wo_ud of layers {EDITED.start}..{EDITED.stop - 1}")
        h = hashlib.sha256()
        f.seek(sec1[0])
        left = sec1[1]
        while left:
            chunk = f.read(min(left, 64 << 20))
            h.update(chunk)
            left -= len(chunk)
        idx = next(i for i, e in enumerate(entries) if e[3] == 1)
        f.seek(16 + 64 * idx + 16)
        f.write(h.digest())
    tmp.replace(out)
    print(f"wrote {out} (Section 1 sha256 {h.hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
