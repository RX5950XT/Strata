# CPU 計算與 RAM 供料的分離實測

2026-10-14，Ryzen 7 5700X（8 核心／16 執行緒）、96 GiB DDR4-2133、Windows 11、orca IQ3_XXS，CUDA 主卡 RTX 5060 Ti 16 GB、副卡 RTX 3060 Ti 8 GB。背景程式保持執行，沒有改 BIOS、關閉 Hyper-V 或更動正式 Strata 設定。

使用者補充：**PBO 從本次工作開始前到全部測試結束，一直都是開啟的。** 此狀態由使用者提供，沒有以 BIOS 讀值獨立核對。正常組已是在 PBO 開啟的環境下測得；以下「停用 boost」僅指暫時變更 Windows 電源方案，沒有關閉 BIOS 的 PBO。這不是 PBO 開／關對照，也不能把「開 PBO」列為尚未採用的加速方法。

**實測判斷：CPU expert 路徑以 CPU 端的權重解碼／計算／執行緒交接為主，RAM 供料是另一個明顯限制。不能把「host 路徑很慢」全部叫作 RAM 頻寬不足，也不能說全部是 CPU 的乘加算力。**

判斷以相同權重、相同計算、逐位元相同輸出的對照為準。把工作集縮到 CPU 快取可容納，仍留下約 73–79% 的相鄰 RAM 組耗時；最大的 gate/up 階段在快取組也仍有 0.362／0.427 ms。這定位的是 CPU 側處理的限制，不能將剩餘時間全換算成純算術百分比，更不能直接當成完整解碼的時間占比。

## 同一份計算，只改資料來源

直接連結目前 build 的 `strata_kernels_cpu.lib`、`ggml-cpu.lib` 與 `ggml-base.lib`，使用真正的 `ExpertPool::run_split_multi_native`、IQ3_XXS gate/up、IQ4_NL down，沒有重寫模型運算。

從目前 orca GGUF 的第 0 層讀取四個真實 expert（每個 2,176,000 bytes），複製同樣的四個 expert 到 64 組位置，讓 RAM 組與快取組的運算內容完全相同。固定 activation、2560 hidden／640 FF、7 workers 加 host，沿用 0／2／4／6／8／10／12／14 的 logical processors。每次啟動都檢查快取／RAM 結果逐位元相同且沒有 NaN；快速交錯時每段也再次核對輸出。

- 快取組：反覆使用同一組 8.70 MB 權重，低於這顆 CPU 的 32 MiB L3。
- RAM 組：輪流使用 557.06 MB 工作集，遠大於 L3；已先觸碰所有頁面，配置與讀檔不放進計時。
- 在同一程序內，每段量 0.35 秒，快取／RAM 順序輪流反轉，每個程序八對。兩次實驗各三輪正常／停用 boost，每個 expert 服務 1、2 tokens 分開量。

下表合併兩次實驗的**正常時脈**結果：每種 token 數各 48 對（六個獨立程序），共 192 個短段。估算 CPU 時脈約 4.23–4.28 GHz。各欄是中位數，中位數之間不能強制相加；「省下時間」以每對相鄰段的比值計算，沒有把短段當成完全獨立的長時間實驗。

| 每個 expert 服務的 tokens | 快取組總 ms／四 experts | RAM 組總 ms／四 experts | RAM 換成快取的配對時間減少 | 快取 gate/up ms | RAM gate/up ms | 快取 down ms | RAM down ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.475 | 0.700 | 27.1% | 0.362 | 0.441 | 0.113 | 0.255 |
| 2 | 0.586 | 0.727 | 21.4% | 0.427 | 0.470 | 0.150 | 0.235 |

gate/up 是最大的階段，而且移除大部分 RAM 讀取後，這段只少約 9–18% 的中位數時間；down 對資料來源的反應較大（約少 36–56%）。所以最精確的說法是：**IQ3_XXS gate/up 的 CPU 處理速度較卡；down 有較明顯的記憶體供料限制。** CPU 側耗時還包含快取存取、排程與執行緒交接，沒有再把這些拆成 ALU、SIMD、lookup、barrier 的百分比。

主證據：[paired-summary.json](paired-summary.json)、[paired-runs.json](paired-runs.json)、[boost-off-paired-runs.json](boost-off-paired-runs.json)、[cpu_ram_bench.cpp](cpu_ram_bench.cpp)。

## 讓 GPU 占用 RAM 供料，保持 CPU 的計算不變

用 5060 Ti 的 host-to-device copy 反覆讀取另一塊 512 MiB pinned RAM，不執行模型運算。copy 程序 pinned 在 logical processor 1，使用 blocking CUDA 同步。三輪控制／DMA 順序交錯，每輪量快取／RAM 的 1、2 tokens，以及純 RAM 讀取；DMA 確認全程活著，沒有提前結束。

| 工作 | 沒有 DMA | GPU 同時搬資料 | 變化 |
| --- | ---: | ---: | ---: |
| 1 token，快取組 ms／四 experts | 0.477 | 0.619 | +29.8% |
| 1 token，RAM 組 ms／四 experts | 0.605 | 1.203 | +98.7% |
| 2 tokens，快取組 ms／四 experts | 0.602 | 0.741 | +23.1% |
| 2 tokens，RAM 組 ms／四 experts | 0.732 | 1.146 | +56.5% |
| 純 RAM 讀取 GB/s | 23.00 | 14.22 | -38.2% |

DMA 程序最多只用了單一 logical processor 約 2.6% 的 CPU 時間；每次 512 MiB transfer 是已知工作量，平均 host-read payload 約 12–13 GB/s，細節見各輪 JSON。這個對照證明 RAM／供料路徑的競爭會拖慢 CPU experts，尤其是 RAM 組。快取組也受影響，因此不把全部差距都算成 DRAM controller 頻寬，或換算成正式解碼一定能提升多少。

這是刻意增加供料競爭的診斷，不是正常解碼的真實總 RAM 流量，也不能證明原本每一刻都已用滿 RAM 頻寬。

證據：[contention-summary.json](contention-summary.json)、[contention-runs.json](contention-runs.json)、[contention-r0-dma-load.json](contention-r0-dma-load.json)、[contention-r1-dma-load.json](contention-r1-dma-load.json)、[contention-r2-dma-load.json](contention-r2-dma-load.json)、[ram_dma_load.cpp](ram_dma_load.cpp)。

## CPU 時脈與硬體 counters

複製原本 Balanced 電源方案，僅在複本停用 Windows boost，BIOS PBO 全程維持開啟；正常組估算約 4.2–4.6 GHz，Windows 停用 boost 組約 3.375 GHz。試過最高時脈 2300 MHz、最高處理器狀態 70%，Windows counter 仍顯示約 3.375 GHz，所以**沒有將它們當成成功的 2.3 GHz 或 70% 實際時脈測試**。raw label 的 `cap2300`／`max70` 是請求名稱，應以記下的時脈讀值為準。

原先每個條件各三秒的交錯測試有 45 筆，變動較大；後續改用上述同一程序內的快速配對作為主要比較。兩批快速配對合計 384 段，正常／低時脈各半，所有輸出核對通過。這些時脈是 Windows Processor Information 的 `% Processor Performance` 乘 base MHz 的估算，沒有宣稱是每核心獨立感測器。部分原先三秒 read 組沒有足夠時脈取樣，結果明確留 `null`。

AMD uProf 5.3.521.0 的 IBS 實際報 `IBS counters are not available`；Core PMC 可以讀取。獨立六筆診斷的 1-token 快取／RAM 組，`DC Fills From Memory (pti)` 約 0.252／1.287，符合資料來源有改變。這是系統範圍、每千指令的 refill counter，**不是完整機器的 DRAM GB/s**；其他程式也包含在 counters 裡。

uProf 的程式啟動模式沒有正常結束，後續改為獨立限時收集並清理本次殘留 counters。硬體 counter 診斷與主要速度計時分開，沒有把 profiler 的耗時混進快速配對結果。IBS 限制也符合 [AMD 的 profile 說明](https://docs.amd.com/r/en-US/63856-uProf-release-notes/CPU-Profiling)；沒有為了取樣重開機或更動 Hyper-V。

證據：[counter-runs.json](counter-runs.json)、[r0-normal-hot-c8-t1.csv](r0-normal-hot-c8-t1.csv)、[r0-normal-ram-c8-t1.csv](r0-normal-ram-c8-t1.csv)、[micro-summary.json](micro-summary.json)、[runs.json](runs.json)、[metadata.json](metadata.json)。

## 完整 100k／200k 解碼確認

使用同一個私有的正式 binary 與 vision encoder，固定主／副卡 1780／2499 槽，停用 adaptive swaps，其他沿用 config，以固定 residency 比較正常／停用 boost。每個長度先完整預填暖機，然後交錯各三次 512-token 解碼；temperature 0、seed 123。原 config 沒有改寫，使用完只關閉本次建立的 engine／vision。

| 實際輸入 tokens | 正常解碼中位數 tok/s | 停用 boost 中位數 tok/s | 正常時脈估算 GHz | 停用 boost GHz |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 32.15 | 31.18 | 4.59 | 3.38 |
| 200,000 | 33.15 | 30.79 | 4.58 | 3.38 |

**這組不作為 CPU／RAM 分離的因果證據。** 不同回合的輸出 hash、MTP 接受情況與 CPU expert 數量不同，100k 正常組也有 27.34–36.75 tok/s 的大幅波動。不能把速度差全當成時脈效果，也不能從它推算換 CPU 的增益。這組固定 cache／停用 swaps 的診斷配置，也不是前一輪正常 `auto` 設定的速度保證。

證據：[decode-clock-summary.json](decode-clock-summary.json)、[decode-clock-runs.json](decode-clock-runs.json)、[decode-clock-telemetry.json](decode-clock-telemetry.json)、[decode-clock-runtime.json](decode-clock-runtime.json)、[decode-clock-engine.log](decode-clock-engine.log)。

## 驗證與還原

- `.panel/build_cpu_ram_bench.bat`／`.panel/build_ram_dma_load.bat`／`.panel/build_cpu_ram_paired.bat`：Windows MSVC 建置成功；連結本機既有 CPU libraries，DMA 連結 CUDA 13.3 runtime。CUDA headers 有既有字碼頁 warning，沒有編譯錯誤。
- 原先 45 筆 timing、30 筆 DMA／control、兩批 384 個 paired 段與完整模型的 14 筆請求都已完成；數量包含暖機／診斷，不能當成獨立完整長測次數。
- 主證據所有 compute 結果逐位元一致，沒有 NaN；模型輸出 hash 不同，因此沒有冒充模型逐 token parity。
- 設定檔與 CPU library 的 hash 與開始前相同，原 Balanced 電源模式已還原，測試用的電源方案複本已刪除；私有 engine、vision、DMA、CPU 測試程序都已結束。原先 panel／正式 server 在本輪開始前已停止，本輪沒有另外把它啟動成常駐服務。[最後核對](final-runtime.json)。

未取得 DRAM controller 的完整實體流量、純算術／lookup／執行緒交接的個別 cycle 占比，未驗證硬體升級能快多少，也未改變正式預設或引擎程式碼。這份結果只定位這台機器目前 IQ3_XXS 的 CPU expert 處理與供料限制；不能當成其他 CPU、量化格式或所有工作負載的結論。
