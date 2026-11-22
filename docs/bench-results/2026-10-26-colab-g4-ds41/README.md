# DeepSeek-V4.1-Flash Uncensored IQ2_XXS on a Colab G4 (2026-10-26)

Model: [Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF](https://huggingface.co/Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF)
(4 shards, 138.6 GB, plus its 110.6 GB Engram table), with Unsloth's DSpark drafter (`dspark-...-Q8_0.gguf`, the MTP) and
vision tower (`mmproj-BF16.gguf`). Engine: the release package `strata-dsv41-0.1.51-linux-x86_64-cuda-cuda13`. Pack: upstream
`4b9e05a2` `tools/dsv41/pack_gguf.py` with the plain recipe (`--engram`, `--engram-meta` = Unsloth's UD-Q8_K_XL shard 1,
`--dspark-gguf`, `--dspark-noise-token 128799 --dspark-markov-rank 256`, `--mmproj`) and `data/dsv41/profile-2.00.bin` as
`expert-profile.bin`.

Machine: Colab G4, RTX PRO 6000 Blackwell (96 GB), AMD EPYC 9B45 (48 threads, AVX-512), 176 GB RAM, 194 GB free disk and an
87 GB `/dev/shm`. The 240 GiB set does not fit on the disk: shards 1 and 4, the drafter and the vision tower live in
`/dev/shm`, with links in `/content/m` (the pack's paths are relative to it).

## Results

Context 204,800 for every run. Greedy, 256 tokens, three prompts (code, English prose, Traditional Chinese). The first round
of each engine start is left out (the expert cache warms up). The same prompt and setting varies up to 1.5x between rounds,
so only large differences count. Raw rows: `bench-5cfg.jsonl` (five starts), `bench.jsonl` (the focused run),
`bench-first.jsonl` (the first start).

| | decode tok/s (median) |
|---|---|
| MTP off (`spec=0`) | 63-66 |
| MTP on, `--dspark on --dspark-conf 0.65` | 87-99 |
| MTP on, `dspark_conf` 0.5 or 0.8, or `dspark_cum` 0.5 | 86-112, within the noise of the line above |
| focused run, one start, 5 rounds: on / conf 0.5 / cum 0.5 | 92.9 / 93.3 / 94.3 |

Load-time settings, each its own start, MTP on: `--experts 5`, `--cpu-threads 24` (default: every thread),
`--dspark-block 3` and `--mmap-experts off` gave no difference beyond that noise. Kept: the defaults plus
`--dspark on --dspark-conf 0.65`. With 96 GB of VRAM about 3% of the expert picks run on the CPU; `--mmap-experts auto`
copies the CPU lane's 49 GB into RAM at load. Load: 4-5.5 minutes.

Prompt reading after warm-up: 16K 1,500 tok/s, 64K 1,913 tok/s. The first long prompt after a start is far slower (72 tok/s
for 16K) while the CPU lane's experts are read from the disk.

Pictures: the engine reads them (`GENI`, `vision_ds4.py`). A 756 x 588 dog photo (268 span tokens) was described as
"a fluffy white dog, likely a Samoyed". The tower wants sides in multiples of 14. The HTTP server does not send pictures to
this engine yet: only Qwen's encoder path is wired.

Not done: a 200K-token request. One try at 169K tokens stalled at 44K after 880 s while the disk was also uploading to Drive;
it was not repeated.

## Google Drive

Everything is in Drive at `Strata-DS41/` (241 GiB: `m/`, `shared/`, `eng/`, `scripts/`, `logs/`), uploaded with rclone at
about 110 MB/s. The rclone token is in `%APPDATA%\rclone\rclone.conf` (remote `gdrive`) and in WSL
`~/.config/rclone-gdrive.conf`. `ds4-up` (WSL `~/bin`) makes a new G4, copies the token in, and runs `from_drive.sh`, which
pulls the set into the same layout and starts the server on port 8095 (Colab itself holds 8080). `ds4-up` has not been run
on a new machine yet.
