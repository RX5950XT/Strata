"""On the Colab VM: one picture through strata-dsv41 --serve (GENI) and the model's description of it.

    python vision_ds4.py IMAGE [engine flags...]
"""
import os, re, struct, subprocess, sys, pathlib
from PIL import Image

sys.path.insert(0, "/content/Strata/tools/dsv41")
sys.path.insert(0, "/content/Strata/tools")
import smoke  # noqa: E402

EXE, PACK = "/content/eng/strata-dsv41", "/content/m/maya-ds4"
img = Image.open(sys.argv[1]).convert("RGB")
# the tower takes sides in multiples of 14 (42 after the 3x3 merge); ~0.45 MP is above its minimum pixel count
k = (450000 / (img.width * img.height)) ** 0.5
img = img.resize((max(42, round(img.width * k / 42) * 42), max(42, round(img.height * k / 42) * 42)), Image.BICUBIC)
print("picture sent as", img.size)
pics = "/content/pics.dsi"
tok = smoke.tokenizer(pathlib.Path(PACK))
head = tok.encode("<｜begin▁of▁sentence｜><｜User｜>", parse_special=True)
tail = tok.encode("\nDescribe this picture in two sentences. What is the main object?<｜Assistant｜></think>", parse_special=True)
IMG = tok.encode("<｜deepseek_image｜>", parse_special=True)
assert len(IMG) == 1, IMG
with open(pics, "wb") as f:
    f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", len(head), img.width, img.height) + img.tobytes())

env = dict(os.environ, LD_LIBRARY_PATH="/content/eng:" + os.environ.get("LD_LIBRARY_PATH", ""), STRATA_DS4_SLOT_KEEP="0")
p = subprocess.Popen([EXE, "--pack", PACK, "--max-context", "204800", "--eos", "1", *sys.argv[2:], "--serve"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/content/logs/eng-vision.log", "w"),
                     text=True, env=env, bufsize=1)
for line in p.stdout:
    if "vision" in line:
        print(line.strip())
    if line.startswith("READY"):
        break


def geni(n_img):
    ids = head + IMG * n_img + tail
    p.stdin.write(f"GENI 200 temperature=0 {pics} {','.join(map(str, ids))}\n")
    p.stdin.flush()
    out, err = [], None
    for line in p.stdout:
        if line.startswith("T "):
            out.append(int(line.split()[1]))
        elif line.startswith("ERR"):
            err = line.strip()                    # a DONE line follows it
        elif line.startswith("DONE"):
            return err or out
    return err or "engine exited"


r = geni(1)
m = re.search(r"span \((\d+) tokens", r if isinstance(r, str) else "")
if m:
    r = geni(int(m.group(1)))
print("span tokens:", m.group(1) if m else "?")
print("ANSWER:", r if isinstance(r, str) else tok.decode(r))
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=300)
except subprocess.TimeoutExpired:
    p.kill()
