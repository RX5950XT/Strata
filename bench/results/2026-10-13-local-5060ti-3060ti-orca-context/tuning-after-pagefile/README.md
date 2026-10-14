# Windows 解碼調整（2026-10-14）

同一台 Ryzen 7 5700X、96 GiB RAM、RTX 5060 Ti 16 GB + RTX 3060 Ti 8 GB、Windows 11，orca IQ3_XXS、204800 context、vision、MTP、int8 KV。背景程式保持執行。

保留 `STRATA_ABOVE_NORMAL=1`，讓主引擎與 expert worker 自動使用 AboveNormal。引擎預設不變更 Windows 優先權；本機 config 的 `env` 才開啟它。設定為 `0` 也不變更，其他值報錯。程式只改排程優先權，沒有改模型計算。

| 優先權 | 三輪解碼 tok/s | 中位數 |
| --- | --- | ---: |
| Normal | 30.5 / 29.3 / 33.0 | 30.5 |
| AboveNormal | 33.2 / 37.4 / 37.0 | 37.0 |

交錯順序 Normal / AboveNormal / AboveNormal / Normal / Normal / AboveNormal。相同 prompt，100,013 input、256 output、temperature 0、seed 123，先做完整預填與快取暖機，正式六輪都重用 100,008 prompt tokens；解碼中位數提升 21.3%。這是這台有 CPU 競爭的機器上、快取暖後的比較，不能套用到所有電腦或全新請求。

| `pcie_frac` | 三輪解碼 tok/s | 中位數 |
| ---: | --- | ---: |
| 0.40 | 35.7 / 33.9 / 39.2 | 35.7 |
| 0.55 | 34.9 / 34.0 / 35.9 | 34.9 |
| 0.75 | 35.4 / 35.6 / 31.9 | 35.4 |
| 0.90 | 35.5 / 32.9 / 31.8 | 32.9 |

三輪交替正序、反序、正序，保持 AboveNormal，其他設定相同。原本 0.40 最快，沒有更改 config 的搬運比例，也沒有候選值需要追加確認輪。

## 分頁檔與快取

新增 `D:\pagefile.sys`，初始 65,536 MiB、最大 98,304 MiB，Windows 已實際配置 64 GiB；重載模型後仍維持。C 槽保留原本初始 1,000 MiB、最大 40,000 MiB。沒有重新啟動 Windows；原設定備份在本機 `.panel/pagefile-before.json`。這增加的是 Windows commit 額度與磁碟占用，不會增加實體 RAM。

模型重載時 commit 上限 190.6 GiB、剩餘 64.8 GiB。主卡快取從調整前 1,780 槽 / 3.61 GiB 增至 2,018 槽 / 4.09 GiB；副卡仍是 2,499 槽 / 5.06 GiB。主卡未達 auto 預估的 3,362 槽，啟動仍有 CUDA 可用 VRAM 0 MiB 的警告；分頁額度不是唯一限制。完整 100k／200k 請求另記在 [optimized](../optimized/README.md)。沒有單獨隔離分頁檔的速度增益。

## 驗證與原始數據

`build-dual.bat` 已通過 CUDA sm_86／sm_120 編譯。`tools/test_windows_priority.py` 從實際 `main` 擷取 Windows 程式碼，檢查未設定、停用、錯誤值、實際 AboveNormal 與 Windows API 失敗；也檢查它在 worker 分流前執行。正常引擎重載後，兩個行程都實際顯示 AboveNormal。新舊 binary 的 `--version` stdout，在未設定、`0`、`1` 三種情況都逐 byte 相同；未測模型逐 token 相同、HIP／SYCL 或長時間穩定性。

CPU、RAM、GPU、VRAM、PCIe、頻率、溫度、功耗、可用 commit、page reads 與磁碟讀取共 262 次取樣，取樣錯誤 0。PCIe 為 NVML 短窗讀值，memory controller 是活動比例；未量到 DRAM controller 的實際頻寬。

- [decision.json](decision.json)：完整選擇依據
- [runs.json](runs.json)：每輪請求與引擎速度
- [telemetry.json](telemetry.json)：所有硬體取樣
- [startup.log](startup.log)：實際快取、commit 與 VRAM 警告
