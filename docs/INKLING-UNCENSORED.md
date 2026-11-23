# Inkling-2-Mini 去拒答版：這台電腦怎麼跑

量測日期 2026-11-22（200K 一節 11-23）。機器：RTX 5060 Ti 16 GB（主卡）＋ RTX 3060 Ti 8 GB、Ryzen 7 5700X、96 GB RAM，Windows 11，引擎 strata-inkling 0.1.65（本 fork 自編，sm_86＋sm_120，CUDA 13.4）。

## 結論

- 跑的是 `models/inkling2-strata/Inkling-2-Mini-Uncensored-SQ-10.strata`：Strata 官方 SQ-10（專家約 2–3 bit），
  只把第 4–23 層的 attention 輸出矩陣 `wo_ud` 換成 OpenUncensored 去拒答版的權重。Abiray 的 GGUF 就是從這份去拒答版量化的；
  去拒答只改了這 20 個矩陣，專家、看圖、DSpark 草稿器都跟原版一模一樣。
- 兩份設定（面板都看得到）：
  - `configs/strata-inkling-2-mini-uncensored-sq-10-112k.json`：只用主卡，上下文 112K（114,688），KV fp8，看圖與 DSpark 都開。
  - `configs/strata-inkling-2-mini-uncensored-sq-10.json`（預設）：兩張卡，上下文 200K（204,800）。全域注意力的 KV 放在 3060 Ti
    （`--kv-gpu 1`，設定用 `expert_gpu` 讓引擎看到兩張卡），一次讀 2,048 個 prompt token（`--chunk 2048`）。
  - 兩份都在位置超過 40K 時停用投機解碼（`--draft-max-pos 40960`），112K 那份一次讀 1,024 個 prompt token。
- Abiray 的 `Inkling-2-Mini-Uncensored-Q3_K_M.gguf` 也能載入（見下），但只能走慢路徑，不實用。

## 為什麼不直接跑 Abiray 的 Q3_K_M GGUF

1. 它是 llama.cpp 主線（#32694）的命名，Strata 只認 Unsloth 的命名。已在 `engine/src/inkling2/ink_gguf_loader.h` 加主線支援
  （24 層計數、`loop.start/end/count`、`loop_{q,k,v,ffn}_ctrl_*`、每層 `loop_msg_query` 拼回兩張 query 表）；
  拼法用 Unsloth 同一份權重的兩種命名逐位元比對得出。
2. 快的 CUDA 後端只吃 `.strata`；GGUF 只能走 EP 後端。EP 後端每個投影都經主機來回、attention 在 CPU。
  實測（Q3_K_M，GPU 已補 Q3_K／Q5_0／Q5_1）：prompt 7–11 tok/s，解碼 3.6–5.3 tok/s；prompt 一批超過 64 token 會當掉
  （已修 `d_dense_x_` 太小；批次上限要加 `--chunk 64`）。

## 量到的數字

| 設定 | prompt | 解碼 |
|---|---|---|
| bench，4K 合成 prompt，DSpark 關 | 1,352 tok/s | 36.5 tok/s |
| bench，4K 合成 prompt，DSpark 開 | 1,391 tok/s | 72.4 tok/s |
| server，真實中文問答（含隱藏的思考段），DSpark 開 | — | 47–48 tok/s |
| 雙卡 EP（`--gpus 2 --chunk 64`），1K prompt，DSpark 關 | 3.8 tok/s | 0.8 tok/s |

## 長上下文（2026-11-23，引擎改過之後）

原版引擎一過 64K 就變很慢：讀 prompt 時的暫存區每個 token 要約 86 KB（58K 時 5 GB、200K 要 17 GB），
DSpark 也替每個位置留了約 21 KB，顯存不夠時 Windows 把它挪到系統記憶體。fork 改了四件事：

1. 讀 prompt 的 K/V 只放 8 份（不是 24 份），分數暫存區固定 512 MB，一開始就配好，不再邊讀邊重配。
2. DSpark 的兩個滑動視窗層和 `f_rnn` 改成環狀緩衝（只留最近一段），200K 時省 3.5 GB。
3. 放專家之前先替上面這些、全域 KV 留位置；放不下的專家改放主記憶體，而不是讓 Windows 挪顯存。
4. `INK_BENCH_PROGRESS=1` 讓 bench 每 8K token 印一次速度。

同一段 3,000 token 輸入跟原版比：Mean KLD 0.000035、前一名相同 99.93%（GPU 加總順序不同的雜訊）。

bench（合成 prompt 全部真的讀進去，DSpark 開，fp8 KV）：

| `--max-context` | 專家放主記憶體 | 4K 短文解碼 | 讀滿 prompt | 填滿後解碼 |
|---|---|---|---|---|
| 65,536（原版） | 0 | 80.4 tok/s | — | — |
| 65,536（改後） | 0 | 86.7–88.3 tok/s | — | — |
| 114,688（現在的設定） | 0 | 88.4 tok/s | 112,000 token 169 秒（平均 661 tok/s） | 26.6 tok/s |
| 131,072 | 0 | 33.8 tok/s（顯存太緊） | 130,000 token 245 秒 | 26.0 tok/s |
| 204,800 | 725／3,072 | 1K 時 49.3 tok/s | 200,000 token 678 秒（平均 295 tok/s） | 10.0 tok/s |

- 讀 prompt 每 8K 的速度（112K 那次）：開頭 1,300、32K 1,040、64K 629、98K 428 tok/s。
- 200K 關掉 DSpark 更慢：填滿後 6.7 tok/s；短文也只有 6.6 tok/s，因為專家在主記憶體時，一次只算 1 個 token 很吃虧。
- 透過 server 實測（112K 設定）：短答 64–65 tok/s；102K token 的長文 151 秒讀完並找到暗號。
- 填滿後解碼變慢，主要是每產生一個 token 都要讀過整段記憶。DSpark 一次驗證多個 token 時會重讀好幾次，所以 100K 以上 DSpark 幫不上忙。

## 已知問題

- Inkling 沒有關得掉的思考：模板送「Thinking effort level: 0」，中文題仍常先想一段，server 把思考段丟掉。`max_tokens` 太小會只剩空白或碎片，給 1,000 以上。
- 上游的雙卡 EP 後端（`--gpus 2`）比單卡慢約 50 倍，不要用；fork 的 `--kv-gpu 1` 是另一種分法（見下面 200K 一節）。

## 重做一次

```sh
python -I tools/inkling_uncensor_patch.py models/inkling2-strata/Inkling-2-Mini-SQ-10.strata models/inkling2-strata/Inkling-2-Mini-Uncensored-SQ-10.strata
```

工具先確認第 0–3 層（沒改過）重新編碼後跟官方位元組完全相同，才寫檔；寫完更新 Section 1 的 SHA-256（`--selfcheck --verify-hash` 通過）。

## 200K：物理極限與兩張卡（2026-11-23）

### 硬體實測（不是規格表）

| | RTX 5060 Ti（主卡） | RTX 3060 Ti |
|---|---|---|
| 可用顯存 | 約 14.2 GB（見下） | 約 7.9 GB |
| 讀取頻寬 | 374 GB/s | 426 GB/s |
| BF16 GEMM（FP32 累加） | 42.4 TFLOPS | 35.6 TFLOPS |
| PCIe 4.0 x8（鎖頁記憶體） | 14.3 GB/s | 13.5 GB/s |

兩張卡沒有 peer access，彼此複製經主機 10 GB/s。跨卡的工作錄不進同一個 CUDA graph：peer copy 不能錄，自旋的門鈴會互卡。
跨卡事件一次約 55 µs。Windows：主卡配置超過約 11.5 GB 後，`cudaMemGetInfo` 和 DXGI 預算都會再少約 1.2 GB；
配到約 14.3 GB 時讀取頻寬就掉到 197 GB/s（被挪到系統記憶體），所以主卡真正能用的是約 14.2 GB。

### 模型的帳

- 權重 10.47 GB（主幹 1.87 + 專家 8.6）。每產生 1 個 token 要讀約 3.07 GB 權重。
- 每多 1 個上下文 token：8 個全域注意力 visit 的 fp8 KV 16.9 KB、DSpark 4.1 KB。
- 容量：主卡單獨約可放 11.8 萬 token（實測 112K 是不變慢的上限，吻合）；200K 開 DSpark 約要 16.8 GB，主卡差約 2.6 GB。
  把全域 KV（200K 時 3.4 GB）放到 3060 Ti，3060 Ti 可放約 43 萬 token。
- 200K 解碼的理論上限：權重 3.07 GB＋KV 3.38 GB，單卡約 57 tok/s；KV 由兩張卡同時讀約 75 tok/s；投機解碼理想約 150 tok/s。
- 200K 讀 prompt 的計算量約 3.8 PFLOP（一半是注意力），兩張卡滿載約 50 秒，只用主卡約 90 秒。

### 改了什麼（引擎 `engine/src/inkling2/cuda/`）

1. 讀 prompt 的注意力改成一個融合的 flash attention kernel（`ink_pf_flash.cuh`，sm_80 以上）：分數不寫回顯存，
   一個 block 算 64 個位置乘上共用同一組 K/V 的 3 個 query head，相對位置偏置在 kernel 裡用 FP32 算，
   不再寫出每次 200 MB 的偏置表。乘加用 FP16 輸入、FP32 累加。跟 CPU FP32 的參考比 KLD 0.004285
   （原本的 cuBLAS 路徑 0.004474），比原版還準；16K prompt 的注意力從 4.3 秒降到 1.0 秒（約主卡 BF16 實測峰值的一半）。
   sm_80 以下照舊走 cuBLAS 分段（4,096 key 一段、online softmax，暫存區固定）。
2. `--kv-gpu N`（`ink_remote_attn.cuh`）：8 個全域 visit 的 KV 和注意力在第 N 張卡。主卡在 graph 裡把 q／新 K,V 寫到鎖頁記憶體、按門鈴、
   自旋等回覆；一條 CPU 執行緒在第二張卡上照同一份 `ink_attention()` 算（`ink_attn_dispatch.cuh`），每種（visit、模式、token 數）錄一個
   graph；最後一個小 kernel 直接寫回覆旗標。樹驗證接受的 token 也在第二張卡寫入 KV（`ink_commit_kv`）。
   3060 Ti 閒置時降到 P8：一次小注意力 1.2 ms；加 1 ms 的 keep-warm kernel 後 0.13 ms。
3. 詞表輸出層（201K 列，S6_K）在 16 列以內直接讀 6-bit 權重算（`k_ink_gemv_sk`）。原本每產生 1 個 token 要先分 22 段解成 BF16
   寫回顯存再乘，約搬 2 GB；現在只讀約 360 MB。不投機的解碼 4K 時 41.2 → 60.0 tok/s。投機驗證的 argmax 也改用這條
   （16 列以內），DSpark 解碼 110 → 117 tok/s；1,204 列跟原本的篩選法逐一比對，結果全部相同（`INK_ARGMAX_CHECK=1`）。
4. 全域 visit 的解碼注意力用 64 段 split-K，query 排在 grid 最快的那一維；每個 key 由 16 個 lane 讀（每 lane 8 bytes），
   一次 4 個 key 一起算再做一次 softmax 更新。跟原 kernel 的最大差異約 5e-6。128K 雙卡不投機 32.8 → 34.0 tok/s。
   把段數改成「剛好一波」（57 段）反而變慢（24.1 vs 25.6），所以維持 64。
5. `--draft-max-pos N`：位置超過 N 就不做投機步驟。改完上面幾項之後，不投機變快很多，交叉點約 40K
   （雙卡：32K 48.9 vs 47.9、64K 38.8 vs 42.3、128K 22.5 vs 34.1 tok/s，投機 vs 不投機；單卡：32K 54.4 vs 53.8、48K 50.2 vs 51.6、
   64K 43.5 vs 49.5），兩份設定都用 40,960。
6. 讀 prompt 一次送的 token 數（`--chunk`）：200K 雙卡用 2,048、112K 單卡用 1,024。112K 單卡用 2,048 會讓主卡顯存不夠
   （剩 0 MB，被挪到系統記憶體，讀 prompt 掉到 1,213 tok/s、解碼 9 tok/s），1,024 剩 460 MB。
7. 除錯開關（環境變數）：`INK_VRAM_LOG`、`INK_BENCH_PROGRESS`、`INK_KLD_CHUNKS`、`INK_RA_PROF`、`INK_RA_CHECK`、`INK_MQ_OFF`、
   `INK_PF_FLASH_OFF`（關融合 kernel）、`INK_PF_FLASH_BF16`、`INK_PF_FLASH_CHECK`（兩條路都算再比）、`INK_SK_GEMV_OFF`、
   `INK_ARGMAX_FULL_MAX`、`INK_ARGMAX_CHECK`、`INK_DEC16_OFF`、`INK_DEC_SPLITS`。

量精度時注意：從 `build-inkling/` 直接跑的 exe 會載入 CUDA 13.4 的 cuBLAS，`engine-local/inkling/` 裡的用發行包附的；
同一份原始碼兩邊 KLD 就差 0.0038。要比，先把 exe 放到 `engine-local/inkling/`。

### 量到的（bench，合成 prompt 全部真的讀進去，fp8 KV）

| 設定 | 讀 200K prompt | 200K 填滿後解碼 | 4K 短文解碼（DSpark／不投機） |
|---|---|---|---|
| 原版引擎，單卡 200K（專家 24% 在主記憶體） | 678 秒 | 10.0 tok/s | 49.3 tok/s（1K） |
| 兩張卡，第一版（cuBLAS 注意力） | 364 秒（平均 549 tok/s） | 22.8 tok/s | 106.5／39.3 tok/s |
| 兩張卡，現在 | **185 秒**（平均 1,101 tok/s） | **26.3 tok/s** | 117／60 tok/s（單卡、2K prompt 量的） |

讀 prompt 每 8K 的速度（兩張卡，現在）：開頭約 1,900、64K 1,330、131K 1,000、197K 795 tok/s。
16K prompt（單卡）：1,408 → 2,207 tok/s。單卡 112K 設定：32K 讀 prompt 1,916 tok/s。

透過 server（真實文字，repo 的文件，中間藏一句暗號）：

| 設定 | 長文 | 讀完並回答 | 暗號 |
|---|---|---|---|
| 200K 雙卡（第一版） | 178,435 token | 320 秒 | 答對 |
| 200K 雙卡（現在） | 172,981 token | 152 秒 | 答對 |
| 200K 雙卡（現在） | 60,271 token | 42 秒 | 答對 |
| 112K 單卡（現在） | 101,986 token | 68 秒 | 答對 |

短答（中文、英文程式題）：雙卡 54–76 tok/s、單卡 57–81 tok/s（含讀 prompt）。64K 以內單卡比雙卡快
（64K 不投機：單卡 49.5、雙卡 42.3 tok/s），雙卡只在超過 112K 時才需要。

### 離極限還差多少

- 200K 讀 prompt 185 秒，理論上 50–90 秒。剩下的時間約一半是注意力（在 3060 Ti 上，融合 kernel 約峰值的一半），
  一半是專家和主幹的矩陣乘法（主卡）；兩張卡是輪流做，不是同時做。要同時做，得讓兩段 prompt 交錯在兩張卡上跑（沒做）。
- 200K 解碼 26.3 tok/s，上限約 75。每個 token 約 38 ms：主卡的權重和小 kernel 約 17 ms（只吃到約一半頻寬）、
  3060 Ti 讀 3.4 GB 的 KV 約 20 ms（約 4 成頻寬），加 8 次交接。
- 3060 Ti 只負責全域注意力。把專家也分過去，每層都要跨卡交接（每次約 0.13 ms，每個 token 48 次），解碼會更慢，所以沒做。
