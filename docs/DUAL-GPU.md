# 雙卡研究：第二張卡怎麼幫上忙

研究日期 2026-09-28。目標機器：RTX 5060 Ti 16 GB（主卡，sm_120）＋ RTX 3060 Ti 8 GB（sm_86），兩張都跑在 PCIe 4.0 x8，沒有 NVLink，也不能走 P2P（Windows、WDDM、GeForce）。

## 結論

**把 3060 Ti 當成「第二個 expert cache」，接在 CPU expert pool 的 hook 裡。**

- 5060 Ti 維持現狀：attention、GDN、router、shared expert、MTP、KV、輸出頭，以及它自己的 expert cache。
- 3060 Ti 只放 profile 裡「主卡放不下的下一批」expert（約 6.5 GB，Q2_0 約 4,500 個），也只負責算這些 expert。
- CPU 繼續算兩張卡都沒有的 miss。

預估解碼速度（模型推估，還沒實測，誤差 ±20%）：Q2_0 4K 從 87 到約 100 tok/s（+15%），IQ3_XXS 4K 從 65 到約 77 tok/s（+18%）。長 context 的 miss 更多，增益百分比會更高。Prefill 大致不變。

## 實作結果（2026-09-28）

用法：`setup.py --expert-gpu yes`（或在設定檔加 `"expert_gpu": "<第二張卡 UUID>"`，引擎參數加 `--expert-gpu 1 --expert-gpu-reserve-mib 700`）。引擎要用這份原始碼自己編（`86;120`），現成引擎不認得這個參數。第二張卡不能用時，引擎印一行 `expert GPU ... disabled: <原因>`，照常用單卡跑。

實測（Ryzen 7 5700X、96 GB RAM、IQ3_XXS、262K context 設定、經 `serve/server.py`、每次 400 token）：

| | 輸出 tok/s |
| --- | --- |
| 原本的引擎 | 27.4、29.3 |
| 新引擎，單卡 | 25.9、27.6；另一次 31.3、34.1 |
| 新引擎，雙卡（3060 Ti 放 3,299 個 expert，和看圖共用） | 35.9、32.3；另一次 35.7、38.6 |
| 6K token 的 prompt 之後：單卡 → 雙卡 | 18.5 → 31.0、24.3 → 37.9 |

- 短 prompt 時雙卡快 13–38%（中位數約 +20%）；context 越長、CPU 要算的越多，差距越大。
- 讀 prompt 的速度不變（6K token：單卡 264、雙卡 275 tok/s）。
- 雙卡和單卡的 greedy 輸出 200/200 token 一樣。這台機器的單卡自己跑兩次，本來就會在第 90 個 token 左右分岔。
- 同一份工作量，這台機器前後跑的速度會差到 ±20%，所以上面每一格都是同一段時間內跑的。
- **orca（uncensored）沒有變快**（單卡 26.7、22.8；雙卡 22.3、25.2）。原因是它的 expert 區有 49 GiB，雙卡時只能釘住 35 GiB（見下面第 1 點），而它的 `--pcie-frac 0.40` 需要釘住的記憶體，沒釘住的那 14 層就退回給 CPU 算，抵掉了第二張卡的好處。

實作時踩到的三件事（都量過）：

1. **Windows 上，有兩個 GPU context 時，釘住（pin）超過約 38–39 GiB 的主機記憶體，兩張卡的 `cudaMalloc` 就一起壞掉**，事後放開也救不回來；跟 `Portable`／`Mapped` 旗標、context 建立順序都無關。單卡時 49 GiB 都沒問題。所以開雙卡時，expert 區最多只釘 36 GiB（`--expert-gpu-pin-gib`，預設 36），其餘照引擎原本的方式用 working-set lock 留在 RAM；讀 prompt 時這幾層改走一般複製，實測速度沒變。
2. **3060 Ti 只拿到零碎的工作時，驅動讓它停在 P8（核心 210 MHz、記憶體 405 MHz）**，每層的 kernel 要 0.7 ms。解法是在低優先權的 stream 上反覆送 1 ms 的「保持運作」kernel（`src/kernels/cuda/keep_warm.cu`）。不能用長的：一個 200 ms 的會把正事卡在後面（p90 193 ms）。
3. **kernel 直接透過 PCIe 讀主機記憶體裡的查表、逐個 float 寫回結果（zero-copy），每層會變成上萬個 PCIe 小交易**，比 CPU 還慢。改成一次 H2D 複製查表和輸入、一次 D2H 複製結果；所有 CUDA 呼叫都交給一條固定在第二張卡上的工作執行緒（`ExpertGpu::work`），主執行緒只填表，不再付每層約 0.2 ms 的 launch 成本。

還沒做的：第二張卡的 expert 固定是啟動時從 profile 挑的，不會跟著對話調整；profile 只有 8,000 組排名，第二張卡最多就放主卡剩下的那些；讀 prompt 時第二張卡不幫忙。

## 為什麼接在 pool hook

每一層的 CPU 交接流程本來就長這樣：GPU0 用 doorbell 把 `x` 和 miss 清單寫進 mapped pinned 記憶體，host 執行緒呼叫 `pool(...)` 算完 miss，把結果寫進 `y_miss`，最後設定 `flag`，GPU0 才接著往下跑（`src/core/session.cpp:915`、`:678`；`src/core/verify.cpp` 的 verify window 也是同一種形狀，走 `drive_pool_multi`）。

所以「GPU0 等 host 回答」這個依賴**本來就存在**。第二張卡只是在 `pool` 裡面接走一部分原本由 CPU 做的 miss：

```
pool(x, ids, weights) 裡面：
  1. 把 miss 分成：GPU1 有的 / 都沒有的
  2. GPU1：launch 一個 kernel，直接讀 host 上的 x（zero-copy），把 expert rows 寫進 y_miss（本來就是 portable mapped）
  3. CPU workers 同時算剩下的
  4. 等 GPU1 的 event，然後才 return（之後照舊設定 flag）
```

GPU0 的 graph、token graph、verify graph 完全不用改；GPU0 這邊感覺到的，就只是 CPU 那段變短了。

## 實測：3060 Ti 跑一趟來回多久

用一個小 CUDA 程式量（每種 2,000 次，host 端用 `cudaEventQuery` poll，跟引擎的 doorbell 做法一樣）：

| 3060 Ti | 中位數 | p90 | p99 |
| --- | ---: | ---: | ---: |
| zero-copy kernel，T=1（讀 10 KB、寫 30 KB） | 14.7 µs | 18.9 µs | 29.8 µs |
| zero-copy kernel，T=4（讀 40 KB、寫 120 KB） | 24.9 µs | 32.1 µs | 45.7 µs |
| H2D + kernel + D2H，T=1 | 70.3 µs | 77.3 µs | 128 µs |
| H2D + kernel + D2H，T=4 | 77.6 µs | 99.8 µs | 177 µs |

對照組：Q2_0 4K 時，一個 verify window 裡 CPU 每層大約要花 190 µs（論文表 5：CPU 露出 9.2 ms ÷ 48 層）。GPU1 分走一半的 miss 以後，CPU 還剩約 95 µs；GPU1 自己「約 25 µs 來回 ＋ 幾個 expert 的 GEMV（1.38 MB ÷ 448 GB/s ≈ 3 µs/個）」可以完全塞在 CPU 那段時間裡面。（後來實作證明這個小測試太樂觀：真正的 expert kernel 用 zero-copy 讀查表時，每層會變成上萬個 PCIe 小交易，見上面「實作結果」第 3 點。最後改成一次複製進、一次複製出，並交給專用執行緒。）

另外量到的：CUDA 的 device 0 是 3060 Ti（CUDA 把「較快」的排前面），跟 `setup.py:247` 的註解一致，所以一定要靠 UUID 指定卡的順序。

## 為什麼不選其他做法

| 做法 | 問題 |
| --- | --- |
| Layer split（部分層整段放到 3060 Ti） | 同一個 token 的 48 層一定是依序跑的，放在慢卡上的層會直接變成總時間的一部分。還得拆 session、KV、graph、checkpoint；而且 fused 路徑會把 FFN residual 延到下一層才寫入（`layer.cpp:1111`、`:1249`），切點要特別處理。工程量最大。 |
| Tensor / row split | 每層都要做好幾次 all-reduce，又沒有 P2P；llama.cpp 已經把 row 模式標成 deprecated。 |
| 把 MTP 搬到 3060 Ti（Codex 的建議） | 主卡大約只多出 1 GB 左右能放 expert，但要改 MTP 的每個入口（`generate.cpp` 裡 7 處以上）、RoPE、gr workspace，還有 head 共用的問題。花的力氣差不多，換回來的 VRAM 大概只有前面做法的 1/6。 |
| 3060 Ti 只當倉庫，權重搬回 GPU0 算 | 每輪要多傳零點幾 GB 的資料，比直接在 GPU1 上算慢一個數量級。 |

今天已經有的：vision encoder 放在第二張卡（`setup.py:972`、`serve/server.py:1289`）。這個保留，vision 和 expert 可以共用 3060 Ti。

## 要改的地方

### 引擎（C++/CUDA）

1. **選卡與旗標**：`generate.cpp` 新增 `--expert-gpu <ordinal>`、`--expert-gpu-reserve-mib <N>`。每一次呼叫 GPU1 的地方都要先 `cudaSetDevice`，再切回 GPU0（加一個小小的 device scope）。
2. **第二個 `ExpertCache`**：在 GPU1 上開，容量用 GPU1 的 `cudaMemGetInfo` 扣掉 reserve 來算；內容取 profile 排名在 GPU0 之後的那一批（`generate.cpp:1331-1405` 的 sizing 邏輯可以沿用）。`ExpertCache` 本身已經支援不同大小的 slot（`expert_cache.cpp:142`、`:180`），只是沒有記錄屬於哪張卡。
3. **Pool 分流**：`expert_pool_dispatch` 和 `expert_pool_dispatch_multi`（`include/strata/core/expert_source.hpp:229`、`:236`）先挑出 GPU1 有的 miss，丟給 GPU1 的 stream（kernel 直接讀寫 mapped host 記憶體），剩下的照舊交給 CPU workers，最後等 GPU1 的 event。
4. **Doorbell 記憶體要加 portable**：`layer.cpp:947` 的 `cudaHostAlloc(..., cudaHostAllocMapped)` 要加上 `cudaHostAllocPortable`，GPU1 才能 map 同一塊記憶體。`y_miss` 本來就是 portable（`session.cpp:497`）。
5. **只做一次的 static 狀態改成每張卡各一份**：`fused_gr.cu:318` 的 `static bool attr`、`qsa.cu:531`／`:659`，還有 GPU1 會用到的 expert kernel 裡的 `cudaFuncSetAttribute`。
6. **Cache 換槽**：`generate.cpp:2216`、`:3332` 的 adaptive swap 先只作用在 GPU0；GPU1 第一版用固定 profile 就好。
7. **Prefill 第一版不動**：prefill 把 expert 經 PCIe 串流到 GPU0（`prefill.cpp:629`），GPU1 不參與。

### Python

1. `setup.py:245` `gpu_info()`：spare 如果是 sm_80 以上而且有 6 GB 以上，就當成 `expert_gpu` 候選；在 Images 那個問題附近加一題「第二張卡用來多放 expert？」，預設 yes（`--yes` 也選 yes）。
2. `setup.py:962` config 新增 `expert_gpu`（UUID），並寫出引擎參數 `--expert-gpu 1 --expert-gpu-reserve-mib <512 或 1400>`（vision 也在這張卡上時用 1400）。
3. `serve/server.py:379` `child_env()`：`CUDA_VISIBLE_DEVICES=<主卡 UUID>,<expert_gpu UUID>`，主卡放前面，這樣 ordinal 0 就是 5060 Ti。Vision 行程維持只看一個 UUID。
4. 本機編譯（`setup.py:617`）：`strata` 本體也要編主卡加 spare 兩種架構（`86;120`）。現成引擎已經有 86/89/120，不用重編。
5. 面板：`serve/panel.py:440` 的角色標示加上「專家」（跟「看圖」同一張卡時顯示「看圖＋專家」）。

## 驗收

- 正確性：開 `--expert-gpu` 和不開，greedy 解碼輸出的 token 要一樣（或 logits 差在誤差範圍內）；sm_86 要跑過 expert kernel 的 parity 測試。
- 速度：同一個 prompt、context、sampling，比較單卡和雙卡的 tok/s、`ms_pool`（`session.cpp:921`）、兩張卡的 VRAM 用量。Q2_0 4K 至少要有可重複的淨增益；如果持平或變慢，就照實記錄，不能算完成。
- 失敗保護：GPU1 不存在或初始化失敗時，要自動退回單卡，並在 log 寫清楚原因。
