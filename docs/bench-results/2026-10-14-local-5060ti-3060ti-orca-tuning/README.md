# orca 後續解碼調整實測

2026-10-14，Windows 11、Strata 0.1.43，Ryzen 7 5700X（8 核心／16 執行緒）、96 GiB RAM（2133 MT/s），主卡 RTX 5060 Ti 16 GB、副卡 RTX 3060 Ti 8 GB。模型為 orca IQ3_XXS，沿用雙 GPU、vision、MTP 4、int8 KV 與 204800 context。

本輪沒有找到可穩定保留的新加速設定。MTP、CPU worker 數量、host 核心位置與 adaptive cache 都測過；短提示詞表現較好的 `--host-core last`，回到正常 `auto` 快取後沒有重現加速，已還原。本輪沒有修改引擎原始碼或 binary。

目前保留前一輪已驗證的配置：`STRATA_ABOVE_NORMAL=1`、PCIe 比例 0.40、MTP `spec-min-p=0.5`、預設 7 個 pool workers 加上 host thread、預設 `host-core first`、`expert-cache auto`、adaptive cache 預設 4／96／0.7。D 槽分頁檔仍是初始 64 GiB／最大 96 GiB，C 槽仍是原本 1000／40000 MiB。

## 最後正常設定的完整 100k／200k

恢復原設定、重啟模型後，各測一次。temperature 0、seed 123、輸出 400 tokens；兩筆 prompt 快取重用都是 0。

| 實際輸入 tokens | 輸出 tokens | 預填 tok/s | 解碼 tok/s | 首 token 秒數 | 總秒數 | MTP 接受率 | expert cache 命中率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,013 | 400 | 1053.6 | 36.8 | 95.173 | 105.986 | 74.0% | 52.5% |
| 200,013 | 400 | 1046.0 | 34.4 | 191.540 | 203.132 | 68.0% | 53.8% |

速度取 engine 的 prompt／decode 計時；首 token 與總秒數取串流用戶端計時。這是保留配置的重新量測，不代表本輪調整帶來額外增益。400 tokens 結束時仍在思考，這兩筆不能證明完整答案品質。

實際主卡快取為 1780 槽／3.61 GiB，副卡為 2499 槽／5.06 GiB；啟動完成後主卡 CUDA 可用 VRAM 為 572 MiB。`auto` 一開始估計 3362 槽，實際配置時縮小，不能以估計容量比較速度。

## MTP 門檻

重用長 prompt 快取，256 output、temperature 0、seed 123；0.3／0.5／0.7 交錯各三次。當時主卡快取為 2018 槽，因此這組數字只適合組內比較。

| spec-min-p | 100k 解碼中位數 tok/s | 200k 解碼中位數 tok/s |
| ---: | ---: | ---: |
| 0.3 | 35.6 | 31.5 |
| 0.5（原值） | 37.0 | 32.0 |
| 0.7 | 34.1 | 33.8 |

0.7 在 200k 快 5.6%，但 100k 慢 7.8%，所以保留 0.5。完整三次數字見 [mtp-decision.json](mtp-decision.json)。

## CPU worker 與核心位置

最初 `auto` 重啟後主卡容量從 1780 變成 2015 槽，這段測試不作為比較依據（[auto-capacity-probes.json](auto-capacity-probes.json)）。後續暫時固定 `--expert-cache 1700`，每次確認實際也是 1700 槽／3.45 GiB，其他配置相同。

使用 `tools/calibrate.py` 的前兩個提示詞：Python 合併程式、冰箱原理解釋。各先暖機 128 output，再交錯各測三次、最多 256 output；允許正常提早結束，至少有 128 output。temperature 0、seed 123。表格是各提示詞的解碼中位數。

| pool workers | host-core | 程式提示 tok/s | 解釋提示 tok/s |
| ---: | --- | ---: | ---: |
| 7 | first，初次 | 27.7 | 34.0 |
| 5 | first | 32.0 | 32.5 |
| 4 | first | 30.7 | 29.5 |
| 2 | first | 29.6 | 32.5 |
| 7 | first，重測 | 24.3 | 27.1 |
| 7 | last | 37.8 | 40.0 |
| 7 | first，最後重測 | 33.0 | 36.8 |

5 workers 的兩提示幾何平均雖快 5.1%，解釋提示卻慢 4.4%，沒有保留。`last` 比後面的 `first` 對照幾何平均快 11.6%，因此進入長上下文驗證；這不是最終配置。

`first` 是 host 在 logical processor 0、workers 在 2／4／6／8／10／12／14；`last` 是 host 在 14、workers 在 0／2／4／6／8／10／12。host 也參與工作，因此「7 workers」不是總共只用 7 條執行緒。`aux-cpus` 在這個版本只對 Linux 生效，沒有用它在 Windows 做假比較，也沒有新增 Windows 實作。

## Adaptive cache

使用候選的 7 workers／host `last`、固定 1700 槽，同樣兩個提示詞與三輪，最多 512 output，允許正常提早結束。

| 更新間隔／最多換槽／decay | 程式提示 tok/s | 解釋提示 tok/s | 兩提示幾何平均 tok/s |
| --- | ---: | ---: | ---: |
| 4／96／0.7（預設） | 37.6 | 36.5 | 37.05 |
| 1／80／0.97 | 38.2 | 37.5 | 37.85 |
| 1／160／0.97 | 37.6 | 37.1 | 37.35 |
| 預設重測 | 38.2 | 35.3 | 36.72 |

最快候選相對初次預設只快 2.2%，沒有超過本輪 3% 採用門檻，也有明顯輪次波動，所以保留預設。CPU 與 cache 每次實際數字及參數見 [probes.json](probes.json)。

## 長上下文確認與還原原因

先固定 1700 槽，比較原設定 `first` 與候選 `last`。每組重新載入、完整預填一次，再重用相同 prompt 各解碼三次；temperature 0、seed 123、每次 400 output。

| 配置 | 100k 完整預填 tok/s | 100k fresh 解碼 tok/s | 100k cached 解碼中位數 tok/s | 200k 完整預填 tok/s | 200k fresh 解碼 tok/s | 200k cached 解碼中位數 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原設定 first，1700 槽 | 1046.1 | 37.6 | 38.1 | 1031.3 | 35.6 | 33.9 |
| 候選 last，1700 槽 | 1061.7 | 41.1 | 38.8 | 1051.7 | 35.7 | 36.0 |

cached 中位數在 100k 只快 1.8%、200k 快 6.2%；兩長度幾何平均約快 4.0%。初步沒有通過「兩長度都快 3%」的規則；另以整體平均評估，暫時把候選放回正常 `auto` 做實際確認，結果如下。

| 正常 auto，實際同為 1780 槽 | 100k fresh 解碼 tok/s | 200k fresh 解碼 tok/s |
| --- | ---: | ---: |
| 候選 last | 31.6 | 33.6 |
| 原設定 first，最後重測 | 36.8 | 34.4 |

候選相對原設定分別慢 14.1%／2.3%。各一次且背景負載會波動，不能據此斷言 `last` 一定較慢；但短測增益沒有在日常配置重現，證據不足以保留，所以已完全恢復本輪前的設定。

原設定固定快取的第一次 200k 測試，在取得 panel 狀態時遇到 10 秒 timeout，模型請求本身已成功。腳本先安全還原設定，再改成每次重啟取得一次驗證資訊，重新完成 200k 的 fresh 與三次 cached；已完成的 100k 保留。原先不完整的 200k 結果仍留在 raw runs，沒有用進上述比較。其中一筆不足 6 個硬體取樣，summary 明確標示不足，沒有補造數字。

## 硬體與服務核對

全程共 2994 個約每秒硬體取樣，取樣錯誤為空。最後兩筆完整請求有 106／203 個取樣，解碼區間有 11／12 個。以下 CPU 使用率與 GPU 使用率為解碼區間中位數；RAM／commit 為完整請求的峰值／最低餘裕。

| 正常原設定 | 系統 CPU | 主引擎 CPU | expert worker CPU | RAM 峰值 GiB | commit 最低餘裕 GiB | 主卡 GPU 使用率 | 副卡 GPU 使用率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | 88.6% | 45.7% | 6.3% | 92.0 | 61.5 | 99.0% | 69.0% |
| 200k | 86.9% | 46.4% | 6.2% | 92.2 | 61.2 | 99.0% | 68.5% |

行程 CPU 百分比以整台 16 個 logical processors 為分母。背景程式保持執行，所以這是使用中的電腦，並非隔離實驗。取樣也保留兩張卡的 VRAM、時脈、溫度、功耗、PCIe 與 Windows page-read／disk-read counters。PCIe、memory-controller counters 只作觀察，不當成實際頻寬證明；沒有取得實體 RAM 頻寬。CPU 時脈由 Windows performance counter 估算，不是每核心感測器讀值。

本輪沒有重跑 engine 的分段計時；[先前瓶頸分析](../2026-10-13-local-5060ti-3060ti-orca-context/host-timing/README.md) 是優先權調整前的結果，不能直接當成目前的精確時間比例。

後續已另外完成 [CPU／RAM 分離對照](../2026-10-14-local-5700x-cpu-ram/README.md)：使用逐位元相同輸出的真實 CPU kernels，定位 CPU 側處理為主、RAM 供料次之；不是以 CPU 使用率推定。

最後執行 `python .panel/verify_more_tuning_runtime.py`：本機 config 與本輪開始前備份完全一致，`/health` 為 `ok`，panel 為 `ready`，兩個 strata 行程都是 AboveNormal，安裝的 engine SHA256 與 `build-dual/strata.exe` 相同；D 分頁檔實際配置 65536 MiB，C 的原始上下限不變。`python .panel/summarize_more_tuning.py` 完成 23 筆完整請求的硬體摘要。這兩個本機腳本位於忽略追蹤的 `.panel/`，結果保存在下列檔案。

## 證據與限制

- [final-decision.json](final-decision.json)：最終決定，恢復原設定。
- [final-runtime.json](final-runtime.json)：服務、實際優先權、原配置一致性、binary hash 與分頁檔核對。
- [startup-full-original-auto-recheck.log](startup-full-original-auto-recheck.log)：最後正常設定啟動與實際快取容量。
- [runs.json](runs.json)：149 筆成功請求，包含暖機、無效初步比較與重試，不是 149 次完整長上下文測試。
- [telemetry.json](telemetry.json)、[hardware-summary.json](hardware-summary.json)：原始硬體取樣與完整請求摘要。
- `candidate.json`、`cpu-cache-decision.json`、`retained-decision.json` 都是過程中的候選判斷，已被 `final-decision.json` 取代，不能用來代表目前配置。

未測所有參數組合、回答品質、逐 token 相同性、長時間穩定性或其他硬體／backend。本輪也沒有重編 native engine；使用前一輪已驗證的 CUDA binary。最後正常 auto 各只有一次 fresh 量測，速度仍會受背景 CPU 負載、MTP 接受率、expert 路由與快取暖度影響，不把單次差異當成額外加速承諾。
