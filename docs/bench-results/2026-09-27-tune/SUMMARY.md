# Strata tune, 2026-09-27

Speeds are engine timers from each run's `results.json`: `engine_decode_tok_s` and `engine_prefill_tok_s`. MTP acceptance is the median stored there. A short run's 1K and 8K figures are medians of two repeats. Expert-cache slots, borrowed prompt slots, and free VRAM are the last load recorded in that run's `engine-log-tail.txt`. Peak VRAM is the model GPU then the vision GPU. The harness `results.md` tables use client-side tok/s and are not repeated here.

A change counts only when mean decode of the two short contexts is more than 3% faster. The engine log is one shared file per model, and a tail can still contain the previous run; a chunk line is counted only when it sits on the load that served that run's prompts.

## Final arguments

Unchanged defaults on both models: `--spec=4 --spec-min-p=0.5 --kv=int8 --vram-reserve-mib=700` (the reserve stays when vision is on; `--no-vision` removes it), expert cache `auto`.

### ISTA — `strata-iq3_xxs.json`, MTP `mtp\rt`

200K:

```text
--max-context=204800 --prefill=8192 --mtp-window=65536
```

262K:

```text
--max-context=262144 --prefill=8192 --mtp-window=65536
```

Stage 1 kept only `--max-context=204800` (mean short decode 39.28 vs 34.85 tok/s, +12.7%). The combined draft was slower than that single change. Later pcie fractions 0.30, 0.20, and 0.10 on the 200K args did not beat it. `--prefill=8192` then did 537.6 tok/s at 32K versus 403.6 at 4096, and that run's log has no chunk shrink. `--mtp-window=65536` did 36.1 tok/s at 131K versus 34.7 at the default 32768.

The saved full ladders still use `--prefill=4096`. The 8192-prefill 262K setting was measured at 1K and 8K only (`ista-262k-best`).

### orca — `strata-orca-iq3_xxs-local.json`

Executable: `engine-local\strata.exe`. MTP stays `mtp\rt`.

200K:

```text
--max-context=204800 --prefill=8192 --mtp-window=65536 --pcie-frac=0.40
```

262K:

```text
--max-context=262144 --prefill=8192 --mtp-window=65536 --pcie-frac=0.40
```

## ISTA stage 1 — short decode

Common setup: contexts 1024 and 8192, 400 tokens, vision off, nonce `bench`. Config default is max-context 262144, prefill 2048, spec 4, spec-min-p 0.5.

| label | override | decode 1K / 8K | prefill 1K / 8K | MTP 1K / 8K | slots | borrow | free MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-base | config | 33.45 / 36.25 | 215.45 / 328.25 | 0.748 / 0.747 | 3311 | 972 | 75 |
| ista-pcie040 | pcie-frac 0.40 | 36.15 / 37.50 | 228.90 / 334.25 | 0.786 / 0.771 | 3311 | 972 | 202 |
| ista-pcie075 | pcie-frac 0.75 | 32.00 / 33.70 | 231.65 / 334.60 | 0.738 / 0.748 | 3311 | 972 | 201 |
| ista-pcie090 | pcie-frac 0.90 | 29.40 / 29.45 | 229.05 / 331.55 | 0.758 / 0.710 | 3311 | 972 | 202 |
| ista-spec3 | spec 3 | 39.00 / 35.90 | 224.60 / 333.65 | 0.871 / 0.808 | 3284 | 969 | 250 |
| ista-spec6 | spec 6, spec-min-p 0.4 | 32.75 / 32.05 | 224.25 / 336.90 | 0.634 / 0.638 | 2760 | 961 | 935 |
| ista-spec4-p03 | spec-min-p 0.3 | 35.45 / 34.35 | 223.50 / 335.95 | 0.653 / 0.629 | 3311 | 972 | 202 |
| ista-spec4-p07 | spec-min-p 0.7 | 36.35 / 37.25 | 226.15 / 336.20 | 0.866 / 0.890 | 3311 | 972 | 202 |
| ista-reserve450 | reserve 450 requested; saved args omit it | 37.00 / 36.35 | 229.05 / 336.30 | 0.807 / 0.764 | 3303 | 972 | 216 |
| ista-adapt0 | adapt-every 0 | 31.10 / 32.25 | 227.00 / 334.70 | 0.758 / 0.784 | 3311 | 972 | 202 |
| ista-adapt16 | adapt-every 16 | 33.25 / 35.35 | 220.00 / 333.35 | 0.785 / 0.704 | 2761 | 961 | 1107 |
| ista-workers6 | pool-workers 6 | 35.00 / 37.30 | 224.90 / 335.75 | 0.760 / 0.773 | 3312 | 972 | 200 |
| ista-ctx200k | max-context 204800 | 40.40 / 38.15 | 235.55 / 338.25 | 0.846 / 0.757 | 3833 | 965 | 200 |
| ista-best | pcie 0.40, spec 3, min-p 0.7, workers 6, ctx 200K | 39.65 / 37.25 | 223.65 / 335.20 | 0.935 / 0.914 | 3834 | 965 | 193 |
| ista-best-r2 | same combination | 37.75 / 37.65 | 229.10 / 337.05 | 0.895 / 0.919 | 3833 | 965 | 206 |

Kept result: `ista-ctx200k`. Mean decode 39.28 vs base 34.85 (+12.7%). The two combination runs averaged 38.45 and 37.70.

Extra pcie probe on `--max-context=204800 --prefill=4096 --mtp-window=65536` (not the stage 1 baseline):

| label | pcie-frac | decode 1K / 8K | prefill 1K / 8K | MTP 1K / 8K | slots | borrow | free MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-pcie030 | 0.30 | 31.45 / 34.35 | 219.50 / 441.35 | 0.734 / 0.769 | 3151 | 1785 | 1199 |
| ista-pcie020 | 0.20 | 38.50 / 35.45 | 217.45 / 447.70 | 0.764 / 0.744 | 3825 | 1789 | 202 |
| ista-pcie010 | 0.10 | 35.95 / 36.25 | 220.00 / 451.35 | 0.805 / 0.752 | 3826 | 1788 | 201 |

Reference short decode at 200K with prefill 2048 was 40.40 / 38.15. None of these three beat that.

## ISTA stage 2 — 32K prefill

Args: max-context 204800, 64 generated tokens. One prompt of 32769 tokens.

| label | prefill | prefill tok/s | decode tok/s | MTP | slots | borrow | free MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-prefill1024 | 1024 | 231.6 | 36.7 | 0.863 | 3833 | 548 | 185 |
| ista-prefill2048 | 2048 | 331.1 | 36.4 | 0.891 | 3155 | 962 | 1232 |
| ista-prefill4096 | 4096 | 403.6 | 35.9 | 0.872 | 3802 | 1786 | 246 |
| ista-prefill8192 | 8192 | 537.6 | 37.5 | 0.915 | 3833 | 3424 | 200 |

Kept: 8192. 537.6 / 403.6 is +33.2% versus 4096. That tail has no `chunk A -> B` line.

## ISTA stage 3 — MTP window at 131072 tokens

Args: max-context 204800, prefill 4096, 400 tokens.

| label | window | prefill tok/s | decode tok/s | MTP | slots | borrow | free MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-mtpw16k | 16384 | 429.0 | 34.5 | 0.692 | 3837 | 1788 | 200 |
| ista-mtpw32k | 32768 (default) | 432.1 | 34.7 | 0.662 | 3833 | 1788 | 200 |
| ista-mtpw64k | 65536 | 427.9 | 36.1 | 0.706 | 3825 | 1789 | 202 |

Kept: 65536. Decode 36.1 vs 34.7 (+4.0%), acceptance 0.706 vs 0.662.

## ISTA ladders

Both ladders below use `--prefill=4096 --mtp-window=65536`, vision on, 400 tokens. They are the paired 200K-vs-262K measurement. The recommended list above uses prefill 8192; there is no second full ladder at that chunk size.

`ista-262k-best` (max-context 262144, prefill 8192, mtp-window 65536, vision off) did complete at short context: 1K decode 38.95 tok/s, prefill 220.95; 8K decode 35.40, prefill 450.05.

### 200K — `ista-final-ladder` (max-context 204800)

Safe limit 203888, so the 204800 and 262144 targets were skipped. Slots 3823, borrow 1790, free 199 MiB.

| context | prompt | TTFT s | prefill tok/s | decode tok/s | MTP | peak VRAM MiB | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 4.858 | 214.3 | 35.4 | 0.772 | 15191 / 1191 | 79227 |
| 8192 | 8192 | 18.398 | 447.7 | 38.3 | 0.745 | 15194 / 1191 | 79338 |
| 32768 | 32769 | 73.554 | 446.4 | 40.0 | 0.817 | 15194 / 1191 | 79403 |
| 65536 | 65536 | 147.174 | 446.1 | 38.7 | 0.787 | 15198 / 1191 | 79433 |
| 131072 | 131072 | 302.211 | 434.4 | 33.6 | 0.665 | 15209 / 1191 | 80060 |

A separate near-limit prompt, `ista-200k-probe`, read 203777 tokens: TTFT 496.715 s, prefill 411.0, decode 33.9, MTP 0.669, VRAM 15191 / 1191, RAM 79473.

### 262K — `ista-final-ladder-262k` (max-context 262144)

262144 was measured at the safe limit 261232. Slots 3303, borrow 1790, free 200 MiB.

| context | prompt | TTFT s | prefill tok/s | decode tok/s | MTP | peak VRAM MiB | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 4.858 | 214.9 | 36.0 | 0.764 | 15198 / 1191 | 77989 |
| 8192 | 8192 | 18.660 | 441.1 | 33.4 | 0.758 | 15198 / 1191 | 78082 |
| 32768 | 32769 | 74.076 | 443.4 | 37.2 | 0.782 | 15206 / 1191 | 78126 |
| 65536 | 65536 | 148.727 | 441.4 | 36.5 | 0.793 | 15198 / 1191 | 78242 |
| 131072 | 131072 | 304.923 | 430.5 | 33.2 | 0.670 | 15230 / 1191 | 78708 |
| 204800 | 204800 | 493.636 | 415.5 | 33.9 | 0.716 | 15199 / 1191 | 78488 |
| 261232 | 261232 | 655.139 | 399.3 | 31.9 | 0.667 | 15198 / 1191 | 78624 |

`ista-262k-probe` read 261121 tokens on max-context 262144: TTFT 657.968 s, prefill 397.5, decode 33.7, MTP 0.723, VRAM 15191 / 1191, RAM 79867.

262K works. The 200K configuration's own safe limit is 203888, so that process skips a 262144 target.

### Speed of 262K versus 200K, same prompt, prefill 4096

Percent is 262K decode (or prefill) relative to the 200K ladder.

| context | decode 200K | decode 262K | decode change | prefill 200K | prefill 262K | prefill change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 35.4 | 36.0 | +1.7% | 214.3 | 214.9 | +0.3% |
| 8192 | 38.3 | 33.4 | −12.8% | 447.7 | 441.1 | −1.5% |
| 32768 | 40.0 | 37.2 | −7.0% | 446.4 | 443.4 | −0.7% |
| 65536 | 38.7 | 36.5 | −5.7% | 446.1 | 441.4 | −1.1% |
| 131072 | 33.6 | 33.2 | −1.2% | 434.4 | 430.5 | −0.9% |

At 261232 tokens the 262K run still decoded at 31.9 tok/s. Peak RAM stayed near 78–80 GiB on both ladders.

### ISTA vision

`ista-vision-200k`: 200K args with prefill 8192, 1024-token text prompt plus the test image, 2000 token budget. The image answer stopped at 871 tokens, engine decode 36.6 tok/s, MTP 0.659.

Opening of that answer: "The user wants a detailed description of the image. Let me carefully examine every element."

The description itself names all four items: a solid red circle upper left, a solid blue square upper right, a solid green triangle lower center with the apex up, and black text "STRATA 42" lower right, with the final 2 clipped by the frame. White background, thin black lines on the top and bottom edges.

## Orca stage 5 — short runs

Config `strata-orca-iq3_xxs-local.json`. Contexts 1024 and 8192, 400 tokens, vision off, nonce `bench`, two repeats.

| label | what changed | decode 1K / 8K | prefill 1K / 8K | MTP 1K / 8K | slots | borrow | free MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orca-base | config, max-context 262144, prefill 2048 | 31.75 / 31.15 | 230.75 / 344.75 | 0.835 / 0.787 | 2956 | 772 | 146 |
| orca-ista-best | ctx 204800, prefill 8192, mtp-window 65536, MTP `mtp\rt` | 31.40 / 31.00 | 209.85 / 550.45 | 0.781 / 0.734 | 3319 | 2749 | 241 |
| orca-mtp-orca | same, MTP `mtp-orca\rt` | 30.85 / 29.60 | 223.20 / 453.35 | 0.773 / 0.779 | 2759 | 1426 | 1256 |
| orca-pcie040 | ista-best args plus pcie-frac 0.40 | 33.35 / 32.15 | 214.05 / 557.15 | 0.847 / 0.752 | 3319 | 2749 | 240 |
| orca-pcie075 | ista-best args plus pcie-frac 0.75 | 28.65 / 28.20 | 210.55 / 552.35 | 0.794 / 0.765 | 3370 | 2749 | 134 |
| orca-spec3 | ista-best args plus spec 3 | 33.20 / 30.90 | 209.35 / 548.35 | 0.844 / 0.786 | 3370 | 2749 | 154 |

### MTP

Original MTP mean decode 31.20 tok/s. Abliterated MTP mean 30.23 (−3.1%). Acceptance moved from 0.781 / 0.734 to 0.773 / 0.779. The abliterated run's own load logged `prompt chunk 8192 -> 4096` and borrowed 1426 slots; 8K prefill fell to 453.35 from 550.45. The original MTP stayed.

### Other knobs, on the original-MTP 200K args

`--pcie-frac=0.40` mean decode 32.75 versus 31.20 (+5.0%). 1K +6.2% (33.35 vs 31.40), 8K +3.7% (32.15 vs 31.00). This is the kept extra flag.

`--pcie-frac=0.75` mean 28.43. `--spec=3` mean 32.05, which is +2.7% and misses the 3% bar (8K decode 30.90 vs 31.00). Spec 3 was not combined with pcie 0.40.

## Orca ladders

Vision on, 400 tokens, final args. `--prefill=8192` stayed intact on these loads (borrow about 2750 slots, no chunk line on the load that served the prompts).

### 200K — `orca-final-ladder-200k`

Safe limit 203888. The 204800 target was measured there; 262144 was skipped. Slots 3368, borrow 2749, free 148 MiB.

| context | prompt | TTFT s | prefill tok/s | decode tok/s | MTP | peak VRAM MiB | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 5.255 | 197.9 | 32.8 | 0.798 | 15262 / 1191 | 87001 |
| 8192 | 8192 | 15.052 | 547.3 | 30.8 | 0.751 | 15254 / 1191 | 87023 |
| 32768 | 32769 | 59.151 | 555.7 | 33.0 | 0.729 | 15256 / 1191 | 87254 |
| 65536 | 65536 | 118.651 | 553.5 | 32.0 | 0.749 | 15262 / 1191 | 87454 |
| 131072 | 131072 | 245.294 | 535.5 | 32.7 | 0.726 | 15280 / 1191 | 87769 |
| 203888 | 203888 | 398.557 | 512.6 | 33.6 | 0.766 | 15281 / 1191 | 87951 |

### 262K — `orca-final-ladder-262k`

262144 was measured at 261232. Slots 2955, borrow 2755, free 147 MiB.

| context | prompt | TTFT s | prefill tok/s | decode tok/s | MTP | peak VRAM MiB | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 5.276 | 196.9 | 32.4 | 0.764 | 15264 / 1191 | 87391 |
| 8192 | 8192 | 16.590 | 496.9 | 27.8 | 0.753 | 15257 / 1191 | 87436 |
| 32768 | 32769 | 60.375 | 544.4 | 28.5 | 0.757 | 15259 / 1191 | 87557 |
| 65536 | 65536 | 120.252 | 546.2 | 29.8 | 0.796 | 15255 / 1191 | 87549 |
| 131072 | 131072 | 246.276 | 533.3 | 30.5 | 0.729 | 15269 / 1191 | 87797 |
| 204800 | 204800 | 399.846 | 513.2 | 29.9 | 0.746 | 15263 / 1191 | 87729 |
| 261232 | 261232 | 530.496 | 493.4 | 18.1 | 0.641 | 15292 / 1191 | 91449 |

262K works: 261232 prompt tokens and 400 generated tokens completed. Peak RAM on that request was 91449 MiB. Decode at that length was 18.1 tok/s, against 29.9 tok/s at 204800 in the same run.

### Speed of 262K versus 200K, same prompt

| context | decode 200K | decode 262K | decode change | prefill 200K | prefill 262K | prefill change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32.8 | 32.4 | −1.2% | 197.9 | 196.9 | −0.5% |
| 8192 | 30.8 | 27.8 | −9.7% | 547.3 | 496.9 | −9.2% |
| 32768 | 33.0 | 28.5 | −13.6% | 555.7 | 544.4 | −2.0% |
| 65536 | 32.0 | 29.8 | −6.9% | 553.5 | 546.2 | −1.3% |
| 131072 | 32.7 | 30.5 | −6.7% | 535.5 | 533.3 | −0.4% |

The 200K run's longest prompt is 203888 tokens (decode 33.6, prefill 512.6). The 262K run's 204800-token prompt is a different length (decode 29.9, prefill 513.2).

## Orca vision

`orca-vision` used the 200K final args, a 1024-token text prompt, the test image, and a 2000-token budget. The text prompt generated 2000 tokens at engine decode 27.5 tok/s (prefill 173.4, MTP 0.795). The image answer stopped at 651 tokens, engine decode 26.7 tok/s, MTP 0.711. That load logged `prompt chunk 8192 -> 4096`, 2757 cache slots, 1426 borrowed, 1313 MiB free.

Opening: "The user wants a detailed description of the image. Let me carefully examine what's present."

The description names a solid red circle on the upper left, a solid blue square on the upper right, a solid green triangle lower center-left with the apex up, and black sans-serif text "STRATA 42" on the lower right, with "42" cut by the right edge. White field, thin black lines on the top and bottom edges.

## Odd log lines

On every load sampled:

- `*** WARNING: --expert-cache is enabled and the GPU hit path is NOT CORRECT.` The next lines say generated tokens diverge from a cache-off run (first difference at token 40 at 2.97% hits, token 0 at 54.4%), timing is real, and output is not. Vision text above is under that warning.
- `cudaHostRegister` of the whole expert arena failed; the engine pinned 48 slices instead (39 GiB on ISTA, 49 GiB on orca) and fell back to 4 KB pages because large pages were refused (`VirtualAlloc error 87`, `SeLockMemoryPrivilege`).
- The expert cache is shrunk after allocation when remaining VRAM is under the reserve. ISTA's last free figure is often about 200 MiB. orca's last free figure on the final ladders is 148 and 147 MiB.
- `this CPU has no AVX-512: the expert kernels run on AVX2.`

Chunk shrink `prompt chunk 8192 -> 4096 tokens so its buffers fit in the expert cache`:

- Absent from `ista-prefill8192` (the 537.6 tok/s probe).
- Present on `orca-mtp-orca` and on `orca-vision`.
- Absent from the load that served `orca-pcie040`, `orca-final-ladder-200k`, and `orca-final-ladder-262k`.

No run in this set died with an engine OOM or a harness error. One 200K orca ladder was started before `--pcie-frac=0.40` had been selected and was stopped; its partial logs are in `orca-final-ladder-200k-aborted` and are not in the tables.
