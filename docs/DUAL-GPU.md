# 雙卡研究：第二張卡怎麼幫上忙

研究日期 2026-09-28。目標機器：RTX 5060 Ti 16 GB（主卡，sm_120）＋ RTX 3060 Ti 8 GB（sm_86），兩張都跑在 PCIe 4.0 x8，沒有 NVLink，也不能走 P2P（Windows、WDDM、GeForce）。

## 結論

**把 3060 Ti 當成「第二個 expert cache」，接在 CPU expert pool 的 hook 裡。**

- 5060 Ti 維持現狀：attention、GDN、router、shared expert、MTP、KV、輸出頭，以及它自己的 expert cache。
- 3060 Ti 放 profile 裡「主卡放不下的下一批」expert，並且跟主卡一樣跟著對話換槽、經自己的 PCIe 分擔 miss（實作結果見下）。
- CPU 繼續算兩張卡都沒有、也沒被分走的 miss。

預估解碼速度（模型推估，還沒實測，誤差 ±20%）：Q2_0 4K 從 87 到約 100 tok/s（+15%），IQ3_XXS 4K 從 65 到約 77 tok/s（+18%）。長 context 的 miss 更多，增益百分比會更高。Prefill 大致不變。

## 實作結果（2026-09-28，第二版）

用法不變：`setup.py --expert-gpu yes`，或在設定檔加 `"expert_gpu": "<第二張卡 UUID>"`、引擎參數加 `--expert-gpu 1 --expert-gpu-reserve-mib 700`。引擎要用這份原始碼自己編（`86;120`）。第二張卡不能用時，引擎印一行 `expert GPU ... disabled: <原因>`，照常用單卡跑。

### 架構：第二張卡交給一個獨立的工作行程

引擎發現 `--expert-gpu` 時，會用同一個 `strata.exe` 啟動一個看不到主卡、只看得到第二張卡的工作行程（`strata --expert-gpu-worker ...`，`src/core/expert_gpu.cpp` 的 `Worker`），用 job object 綁住，引擎結束它就跟著結束。expert 區改放在具名的共用記憶體（pagefile section），兩個行程看到同一份，不多佔 RAM。引擎負責所有決定（哪些 expert 放第二張卡、哪些 miss 交給它、換哪些槽），工作行程只負責搬資料和計算，兩邊用共用記憶體裡的計數器交接。

第二張卡現在做的事，和主卡一樣：

| | 主卡 | 第二張卡 |
| --- | --- | --- |
| 快取裡的 expert（解碼） | ✓ | ✓（profile 排名在主卡之後的那一批） |
| 跟著對話換槽（每 4 輪最多 96 個） | ✓ | ✓（同一層內換，和主卡不重複） |
| 從 RAM 經自己的 PCIe 搬 miss 來算（解碼） | ✓（剩下的 55%） | ✓（先拿 30%，`--expert-gpu-pcie-frac`） |
| 讀 prompt 時算 expert | ✓ | —（併入上游 0.1.15 時拿掉，見下） |

### 實測（Ryzen 7 5700X、96 GB RAM、IQ3_XXS、經 `serve/server.py`、每次 400 token）

同一段時間內交錯跑。這台機器同一份工作量前後會差到 ±15%，所以看同一行的比較：

| 設定 | 單卡 tok/s | 雙卡 tok/s |
| --- | --- | --- |
| Qwen IQ3_XXS、262K context（看圖也在第二張卡） | 28.2、29.3 | 31.5、33.8；安裝後另一次 33.1、37.3 |
| orca IQ3_XXS、262K | 25.4、27.6 | 32.6、36.3 |
| orca IQ3_XXS、200K | 24.3、25.7 | 29.8、31.8 |
| `bench/expert_gpu_smoke.py`（8K context、200 token） | 28.4、27.2；另一次 22.2 | 35.0、34.6；另一次 27.0 |

- 解碼：雙卡快 12–32%。orca 這次也變快了：第一版因為鎖定上限只能鎖 35 GiB，沒有增益。
- 讀 prompt：71 token 的第一個請求，首字時間約 2.5–2.6 秒（單卡 2.5–4.5 秒）。
- 雙卡和單卡的 greedy 輸出 200/200 token 一樣（smoke）。單卡讀 prompt 後的 GDN 狀態雜湊，和這次修改前逐位元相同。

### 併入上游 0.1.15（2026-09-28）

上游把讀 prompt 的路徑整段改寫（llama.cpp 的 MMQ、整段串流，長 prompt 約快 2 倍），也加了同樣做法的 AVX2 多 token CPU kernel。合併時：

- 讀 prompt 用上游的新寫法。舊版「第二張卡每層分擔約 40% expert」接不上新寫法，拿掉了（`--expert-gpu-prefill-frac` 已移除）。重新接上最多只省長 prompt 約 10–15% 的時間，卻要改上游正在頻繁更新的核心程式，每次同步都會再衝突。
- CPU 的 AVX2 多 token kernel 用上游的版本，這邊的版本不再保留。
- 解碼時 GPU 每層的改進（IQ 投影一次解碼給所有 token、只讀 CPU 算的那幾列）保留，結果逐位元不變。

同一段時間交錯跑（Qwen IQ3_XXS、16K context，`generate`）：

| | 上游 0.1.15 單卡 | 合併版單卡 | 合併版雙卡 | 合併前雙卡 |
| --- | --- | --- | --- | --- |
| 解碼 300 token（tok/s） | 32.2、32.0、31.8 | 32.3、32.9、32.5 | 43.3、41.2、43.8 | |
| 讀 4457 token 的 prompt（tok/s） | 427、425 | | 470、457 | 345、347 |

- 單卡時合併版和上游的 greedy 輸出 200/200 token 相同。
- 評估過但沒做：hyper-connection 的 `gr_down` / `gr_up`（每輪合計約 2.7 ms）改寫、orca 的 hc 權重改讀原本的 Q6_K/Q8_0。兩者各自最多省約 1–2%，前者還會改變加總順序（結果不再逐位元相同），不划算。

### 量到的問題和解法

1. **鎖定記憶體的上限，是所有 GPU 加起來的總額，約 78 GiB（這台 96 GB 的電腦）**。同一個行程有兩個 GPU context 時，每鎖 1 GiB 會同時算在兩張卡上，所以第一版只能鎖約 39 GiB。改成兩個行程以後，主卡照單卡的方式鎖滿整個 expert 區（Qwen 39 GiB、orca 49 GiB），工作行程用剩下的額度，鎖住前面的層，給自己的 PCIe 讀取用：預設是 RAM 的 3/4，扣掉主卡已鎖的量，再留 2 GiB（Qwen 29.6 GiB、orca 19.7 GiB）。可以用 `--expert-gpu-pin-gib` 指定。量法：兩個行程鎖同一塊共用記憶體，41 + 30 GiB 正常，44 + 34 GiB 在第 78 GiB 時失敗。
2. **沒事做的卡會在 1 秒內掉到 P8，PCIe 也降到 Gen1**。讀 prompt 的那幾秒，第二張卡沒有工作，回覆一開始要花約 1.5 秒爬回全速（每組 expert 一開始 88 µs，熱了以後 7 µs）。現在一個請求一開始，引擎就通知工作行程保溫：每 1 ms 送一個 1 ms 的空轉 kernel，請求結束 1 秒後才停。間隔拉到 15 ms 不夠，約 7 秒後還是會掉速。
3. **WDDM 每次 CUDA 呼叫約 10 µs，而且同一個行程開多條執行緒送也不會更快**。解碼時：
   - 工作行程每層原本要約 12 次呼叫，現在改成一個 CUDA graph，依 (格式, token 數, k) 各錄一份。
   - 主卡每層的 PCIe 搬運（2 次 `cudaMemcpyAsync` 加 1 次 `cudaLaunchHostFunc`，約 80 µs）原本卡在 CPU 開始算之前，改成交給 Verifier 的專用執行緒送出。上游 0.1.14 起預設改用 copy kernel（`--pcie-mode auto`，issue #31），這條執行緒只在 `--pcie-mode dma` 時用到。
4. 第一版留下的 3 件事仍然有效：WDDM 的 `cudaMalloc` 要等第一次寫入才真的佔用記憶體，所以槽位先寫零，再檢查保留量；表格和輸入一次複製進去、結果一次複製回來；不用 zero-copy。

### 還沒做的

- 讀長 prompt 時，現在卡在主卡上 expert 以外的部分（attention、GDN、PLE），這部分第二張卡幫不上。
- CPU 算 expert 仍是解碼的主要瓶頸：RAM 實測只讀得到約 25 GB/s，CPU 和兩張卡的 PCIe 讀取共用這個上限。
- 工作行程只在 Windows 上實作；其他平台的引擎會照常用單卡跑。

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
