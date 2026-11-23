# ISTA benchmark：Stage 1–4

數值只取自各輪 `results.json` 與 `engine-log-tail.txt`。速度均為引擎計時；Stage 1 的 1K／8K 是各跑兩次的中位數。`slots`、`borrow`、`free` 取各輪 log 最後一次模型載入紀錄，單位依序為 cache slots、cache slots、MiB。

## Stage 1：短 prompt decode

| 標籤 | 要求的 override | decode 1K / 8K tok/s | prefill 1K / 8K tok/s | MTP 1K / 8K | slots | borrow | free MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-base | 預設 | 33.45 / 36.25 | 215.45 / 328.25 | 0.75 / 0.75 | 3311 | 972 | 75 |
| ista-pcie040 | `--pcie-frac=0.40` | 36.15 / 37.50 | 228.90 / 334.25 | 0.79 / 0.77 | 3311 | 972 | 202 |
| ista-pcie075 | `--pcie-frac=0.75` | 32.00 / 33.70 | 231.65 / 334.60 | 0.74 / 0.75 | 3311 | 972 | 201 |
| ista-pcie090 | `--pcie-frac=0.90` | 29.40 / 29.45 | 229.05 / 331.55 | 0.76 / 0.71 | 3311 | 972 | 202 |
| ista-spec3 | `--spec=3` | 39.00 / 35.90 | 224.60 / 333.65 | 0.87 / 0.81 | 3284 | 969 | 250 |
| ista-spec6 | `--spec=6 --spec-min-p=0.4` | 32.75 / 32.05 | 224.25 / 336.90 | 0.63 / 0.64 | 2760 | 961 | 935 |
| ista-spec4-p03 | `--spec-min-p=0.3` | 35.45 / 34.35 | 223.50 / 335.95 | 0.65 / 0.63 | 3311 | 972 | 202 |
| ista-spec4-p07 | `--spec-min-p=0.7` | 36.35 / 37.25 | 226.15 / 336.20 | 0.87 / 0.89 | 3311 | 972 | 202 |
| ista-reserve450 | `--vram-reserve-mib=450` | 37.00 / 36.35 | 229.05 / 336.30 | 0.81 / 0.76 | 3303 | 972 | 216 |
| ista-adapt0 | `--adapt-every=0` | 31.10 / 32.25 | 227.00 / 334.70 | 0.76 / 0.78 | 3311 | 972 | 202 |
| ista-adapt16 | `--adapt-every=16` | 33.25 / 35.35 | 220.00 / 333.35 | 0.79 / 0.70 | 2761 | 961 | 1107 |
| ista-workers6 | `--pool-workers=6` | 35.00 / 37.30 | 224.90 / 335.75 | 0.76 / 0.77 | 3312 | 972 | 200 |
| ista-ctx200k | `--max-context=204800` | 40.40 / 38.15 | 235.55 / 338.25 | 0.85 / 0.76 | 3833 | 965 | 200 |
| ista-best | `--pcie-frac=0.40 --spec=3 --spec-min-p=0.7 --pool-workers=6 --max-context=204800` | 39.65 / 37.25 | 223.65 / 335.20 | 0.94 / 0.91 | 3834 | 965 | 193 |
| ista-best-r2 | 同上複測 | 37.75 / 37.65 | 229.10 / 337.05 | 0.89 / 0.92 | 3833 | 965 | 206 |

**選擇：`ista-ctx200k`。** 兩個 context 的 decode 平均為 39.28 tok/s，對 `ista-base` 的 34.85 快 **12.7%**。兩次合併測試的平均各為 38.45、37.70，均低於最佳單因子。`ista-reserve450` 雖有數字改善，但 harness 在 `--no-vision` 時移除了該旗標，不能歸因於 reserve 設定。

## Stage 2：32K prefill

| 標籤 | prefill override | prefill tok/s | decode tok/s | MTP | slots | borrow | free MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-prefill1024 | `--prefill=1024` | 231.6 | 36.7 | 0.863 | 3833 | 548 | 185 |
| ista-prefill2048 | 預設 2048 | 331.1 | 36.4 | 0.891 | 3155 | 962 | 1232 |
| ista-prefill4096 | `--prefill=4096` | 403.6 | 35.9 | 0.872 | 3802 | 1786 | 246 |

**選擇：`--prefill=4096`，比 2048 快 21.9%。** 4096 的 log 未見 chunk 對半縮小。

## Stage 3：131K MTP window

| 標籤 | MTP window | prefill tok/s | decode tok/s | MTP | slots | borrow | free MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-mtpw16k | 16384 | 429.0 | 34.5 | 0.692 | 3837 | 1788 | 200 |
| ista-mtpw32k | 預設 32768 | 432.1 | 34.7 | 0.662 | 3833 | 1788 | 200 |
| ista-mtpw64k | 65536 | 427.9 | 36.1 | 0.706 | 3825 | 1789 | 202 |

**選擇：`--mtp-window=65536`，比預設 32768 的 decode 快 4.0%。** MTP acceptance 也較高。

## Stage 4：vision 開啟的 context ladder

使用 `--max-context=204800 --prefill=4096 --mtp-window=65536`。Stage 4 是容量驗證，沒有參數競賽或勝者。

| 目標 context | 實際 prompt tokens | TTFT s | engine prefill tok/s | engine decode tok/s | MTP | peak VRAM GPU0/GPU1 MiB | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 4.858 | 214.3 | 35.4 | 0.772 | 15191 / 1191 | 79227 |
| 8192 | 8192 | 18.398 | 447.7 | 38.3 | 0.745 | 15194 / 1191 | 79338 |
| 32768 | 32769 | 73.554 | 446.4 | 40.0 | 0.817 | 15194 / 1191 | 79403 |
| 65536 | 65536 | 147.174 | 446.1 | 38.7 | 0.787 | 15198 / 1191 | 79433 |
| 131072 | 131072 | 302.211 | 434.4 | 33.6 | 0.665 | 15209 / 1191 | 80060 |
| 204800 | skipped：安全上限 203888 | — | — | — | — | — | — |
| 262144 | skipped：安全上限 203888 | — | — | — | — | — | — |

harness 會保留 400 個生成 tokens 與 512 個安全 tokens，因此清單中的上限目標會被跳過。為查證容量，另外各跑一次上限內 prompt；沒有重跑失敗設定：

| 標籤 | max-context | 實際 prompt tokens | TTFT s | engine prefill tok/s | engine decode tok/s | MTP | peak VRAM GPU0/GPU1 MiB | peak RAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ista-200k-probe | 204800 | 203777 | 496.715 | 411.0 | 33.9 | 0.669 | 15191 / 1191 | 79473 |
| ista-262k-probe | 262144 | 261121 | 657.968 | 397.5 | 33.7 | 0.723 | 15191 / 1191 | 79867 |

**262K 可運作**：`--max-context=262144` 的單次請求成功讀入 261121 tokens 並生成 400 tokens。最終較快的 200K 設定本身無法接受 262K prompt。

## Log 與輸出限制

- 引擎明示 expert cache 的 GPU hit path **不正確**：輸出會偏離 cache-off 結果；log 說計時有效、文字輸出無效。這影響所有測試的回答可信度。
- `ista-base` 的最後一次載入僅剩 75 MiB VRAM，log 標為 `LOW`。多輪載入都因可用 VRAM 不足縮小 expert cache；Stage 4 最後剩 199 MiB，prompt path 借用 1790 slots。未見 `chunk A -> B` 或 chunk 對半縮小紀錄。
- expert arena 的整體 `cudaHostRegister` 失敗，改為分片 pin 39 GiB；大頁面因權限不足而退回 4 KB 頁面。
- vision 結果檔只保存回答前 200 字，這段沒有提到紅圓、藍矩形、綠三角或 `STRATA 42`；加上引擎的輸出正確性警告，**圖片辨識未獲驗證**。
