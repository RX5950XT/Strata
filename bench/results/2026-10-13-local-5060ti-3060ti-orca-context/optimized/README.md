# 保留設定後的完整 100k／200k 測試

2026-10-14，同一台 Ryzen 7 5700X、96 GiB RAM、RTX 5060 Ti 16 GB + RTX 3060 Ti 8 GB、Windows 11，orca IQ3_XXS。雙 GPU、vision、MTP、int8 KV、204800 context；主引擎及 worker 使用 `STRATA_ABOVE_NORMAL=1`，`--pcie-frac 0.40`。D 槽新增初始 64 GiB／最大 96 GiB 分頁檔，C 槽保留原設定。詳見 [交錯調整測試](../tuning-after-pagefile/README.md)。

| 目標輸入 | 實際輸入 | 輸出 | 預填 tok/s | 解碼 tok/s | TTFT 秒 | 總秒數 | MTP 接受率 | RAM 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 100,013 | 400 | 1,045.9 | 35.9 | 95.805 | 106.899 | 0.675 | 93,687 |
| 200,000 | 200,014 | 400 | 1,031.6 | 31.9 | 194.219 | 206.715 | 0.645 | 93,778 |

沿用 `bench/ctx_bench.py` 的 repository corpus、原 nonce 與 request 方法（temperature 0.6、top_p 0.95），各測一次；100k 接著 200k，沒有重載，兩筆 log 與 API usage 都確認 reused 0。引擎計時與 client 計時分開保存。模型生成的 400 tokens 都仍在 thinking，這是吞吐測試，不是完整回答品質驗證。

原始 2026-10-13 同機測量為 26.3／24.8 tok/s，本次較高；兩日外部負載、快取與 sampling 不完全相同，不能把差額全部歸給某個調整。優先權的單獨增益由同輪交錯測試支持：Normal 30.5 → AboveNormal 37.0 tok/s，中位數增加 21.3%。未單獨隔離分頁檔的速度增益。

## 硬體取樣

共 328 次取樣，沒有取樣錯誤；兩筆請求 RAM 峰值約 91.5／91.6 GiB，可用 commit 最低 60.82／60.79 GiB。D 槽分頁檔測後仍為 65,536 MiB，兩個引擎仍為 AboveNormal，`/health` 回覆 `ok`。

以下為解碼期間的中位數，100k／200k 分別有 11／12 筆取樣；完整請求及所有取樣的最低／最高值在 `summary.json`。

| 指標 | 100k | 200k |
| --- | ---: | ---: |
| 全機 CPU busy | 90.1% | 90.1% |
| 主引擎 CPU，占全機 | 45.7% | 45.1% |
| worker CPU，占全機 | 5.4% | 5.2% |
| CPU 頻率估計 | 4.62 GHz | 4.61 GHz |
| 可用 RAM | 4.59 GiB | 4.52 GiB |
| 主卡 busy / memory controller 活動 | 98% / 24% | 99% / 26% |
| 副卡 busy / memory controller 活動 | 100% / 4% | 100% / 4% |
| 主卡 VRAM | 13,484 MiB | 13,484 MiB |
| 副卡 VRAM | 6,751 MiB | 6,751 MiB |
| 主卡 core / memory 時脈 | 2,640 / 14,001 MHz | 2,640 / 14,001 MHz |
| 副卡 core / memory 時脈 | 1,995 / 6,801 MHz | 1,995 / 6,801 MHz |
| 主卡溫度 / 功耗 | 53°C / 75.0 W | 54.5°C / 75.2 W |
| 副卡溫度 / 功耗 | 55°C / 74.0 W | 55°C / 73.4 W |
| 主卡 PCIe RX / TX | 3,930 / 390 MiB/s | 3,853 / 621 MiB/s |
| 副卡 PCIe RX / TX | 1,501 / 830 MiB/s | 2,703 / 662 MiB/s |
| Page Reads/sec | 15.5 | 2.9 |

GPU busy 包含等待，memory controller 是活動比例，NVML PCIe 是短窗讀值，不能據此宣稱某條頻寬已用滿。CPU 頻率由 Windows base frequency × performance ratio 估算。仍未取得 DRAM controller 實際頻寬；本輪沒有再開引擎分段計時，先前瓶頸分析見 [host-timing](../host-timing/README.md)。

啟動仍有 CUDA 可用 VRAM 0 MiB 的警告，兩筆長請求均完成；未測長時間穩定性、其他硬體、HIP／SYCL 或模型逐 token 一致性。

- [runs.json](runs.json)：API 與引擎計時
- [summary.json](summary.json)：完整請求／解碼期間硬體統計
- [telemetry.json](telemetry.json)：原始 CPU、RAM、GPU、VRAM、PCIe、頻率、溫度、功耗、commit、page reads 與磁碟讀取
- [100000.log](100000.log)、[200000.log](200000.log)：引擎完整請求紀錄
- [retained-settings.json](retained-settings.json)：測後服務、行程優先權、分頁檔與已安裝 binary 核對
