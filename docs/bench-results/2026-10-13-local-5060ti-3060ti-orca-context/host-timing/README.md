# 解碼瓶頸量測（2026-10-14）

主要時間花在 host expert 路徑：CPU 處理 RAM 裡的 expert、整理工作與等待第二張卡，共占解碼時間約 57–61%。其中扣掉副卡等待後，CPU expert pool 與結果複製占 44–49%，是最大的單項。

| 每個解碼 window | 100k prompt | 200k prompt |
| --- | ---: | ---: |
| 本輪解碼 tok/s | 27.0 | 27.2 |
| 解碼 ms/window | 75.72 | 88.52 |
| CPU expert pool + RAM + 結果複製 | 33.38 ms / 44.1% | 43.36 ms / 49.0% |
| CPU 工作完成後等待副卡 | 6.05 ms / 8.0% | 6.51 ms / 7.4% |
| 等待主卡推進 | 20.81 ms / 27.5% | 22.50 ms / 25.4% |
| MTP draft | 6.48 ms / 8.6% | 6.98 ms / 7.9% |
| 整條 host 路徑（包含前兩項及排程等） | 43.35 ms / 57.3% | 54.09 ms / 61.1% |

後面一列包含前面幾列，不能再次相加。`CPU` 原計時包住 CPU expert pool 與 `ExpertGpu::finish()`；本次使用既有的 `ExpertGpu::ms_wait` 扣除真正的副卡等待，剩下包含 CPU 運算、RAM 存取與副卡結果的 memcpy。這不是 DRAM controller 頻寬量測，不能再由它直接分成 CPU 算力和 RAM 頻寬。

## 限制快取與 CPU 的因素

- 主卡 auto 預估 3,362 槽，寫入後三次報 `only 0 MiB free` 而縮到 1,780 槽（3.61 GiB）；副卡 2,499 槽（5.06 GiB）。兩筆 decode cache hit rate 約 52%，很多 expert 還需交給 host。
- 本輪 decode 時可用 Windows commit 中位數僅 1.55 / 1.93 GiB（100k / 200k）。分頁檔設定為初始 1,000 MiB、最大 40,000 MiB。引擎自己的 Windows 訊息指出 GPU 配置也需要 page-file 額度；解除這個限制、確認主卡能保留更多 cache 是值得驗證的調整，尚未實測其增益。
- 全機 CPU busy 中位數約 87%；主引擎約 39 / 43%，worker 約 5%。用同時取樣的差值估計，其他系統工作占約 44 / 40%。另一次兩秒 process 取樣看到 Blue Archive、AxonDeck 和背景 Python 合計約 35%，顯示有 CPU 競爭；未停止這些工作。
- CPU 頻率估計約 4.6 GHz；RAM 的 ConfiguredClockSpeed 為 2133。沒有 DRAM controller 實際頻寬讀值，不能宣稱 RAM 頻寬已用滿。
- 主卡 GPU busy 中位數 98%，記憶體控制器活動比例僅 24 / 19%；副卡分別 63% 與 3%。主卡約 49 / 52°C、副卡約 51 / 55°C，時脈穩定。GPU busy 含等待期間，這些讀值不能當成顯存頻寬已飽和的證據。
- NVML PCIe RX 中位數：主卡約 2.39 / 2.65 GiB/s，副卡約 0.70 / 3.82 GiB/s。這是每次短取樣窗的讀值，存在突發傳輸，不能據此排除 PCIe 延遲；副卡對關鍵路徑的真正等待則已由引擎量為 7–8%。

## 方法與有效範圍

與前次相同的 orca、204800 context、雙 GPU、vision、MTP、int8 KV 設定；同一份 source corpus，prompt 100,013 / 200,014 tokens，輸出各 400，reused 0，各測一次。開啟 `STRATA_DECODE_TIMING=1`，關閉 GPU event 的 `STRATA_VERIFY_PROFILE`。使用只額外列印既有 worker 計時器的測試執行檔，沒有改運算或路由；編譯後原始碼已還原，測完設定與正常引擎也已恢復。

先前完整 GPU stage profile 放在 `../profile/`。100k 當時 CPU busy 約 93%、commit 僅約 0.74 GiB，解碼 15.7 tok/s；200k 資源餘裕回升後為 24.3 tok/s。GPU event 和外部負載會改變時間，這份結論以後續輕量計時為主。本輪 27 tok/s 是重新量測結果，尚未套用效能調整，不能視為改善已達成。

取樣涵蓋 CPU、引擎行程 CPU、GPU、VRAM、PCIe、頻率、溫度、功耗、可用 RAM/commit、page reads 與磁碟讀取；100k / 200k 解碼區間各有 15 筆取樣，沒有取樣錯誤。原始數據：[summary.json](summary.json)、[runs.json](runs.json)、[telemetry.json](telemetry.json)、[100000.log](100000.log)、[200000.log](200000.log)、[startup.log](startup.log)。
