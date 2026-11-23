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
  - 兩份在任何長度都開投機解碼（DSpark），112K 那份一次讀 1,024 個 prompt token。
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
5. `--draft-max-pos N`：位置超過 N 就不做投機步驟。第二輪時長文投機比較慢，設定用過 40,960；第三輪（下面 8、9）修好之後
   投機在每個長度都比較快，設定已拿掉這個參數（參數還在，預設不限制）。
6. 讀 prompt 一次送的 token 數（`--chunk`）：200K 雙卡用 2,048、112K 單卡用 1,024。112K 單卡用 2,048 會讓主卡顯存不夠
   （剩 0 MB，被挪到系統記憶體，讀 prompt 掉到 1,213 tok/s、解碼 9 tok/s），1,024 剩 460 MB。
7. 除錯開關（環境變數）：`INK_VRAM_LOG`、`INK_BENCH_PROGRESS`、`INK_KLD_CHUNKS`、`INK_RA_PROF`、`INK_RA_CHECK`、`INK_MQ_OFF`、
   `INK_PF_FLASH_OFF`（關融合 kernel）、`INK_PF_FLASH_BF16`、`INK_PF_FLASH_CHECK`（兩條路都算再比）、`INK_SK_GEMV_OFF`、
   `INK_ARGMAX_FULL_MAX`、`INK_ARGMAX_CHECK`、`INK_DEC16_OFF`、`INK_DEC_SPLITS`、`INK_DEC_TC_OFF`、`INK_DSPARK_TC_OFF`、
   `INK_LANES_OFF`、`INK_SPEC_PROF`（投機每一步各段的時間）、`INK_KLD_DECODE_FROM=N`（KLD 從第 N 個 token 起逐個解碼，量解碼路徑）、
   `INK_PF_PROF`（讀 prompt 每一段的時間，要配 `INK_LANES_OFF`）、`INK_MOE_TR128_OFF`（專家改回 64 列一塊）。

第三輪（同一天稍晚）：

8. 全域 visit 的解碼和樹驗證注意力改用 tensor core（`ink_dec_tc.cuh`，sm_80 以上、fp8 KV、16 個 query 以內）。
   原本每個 query 各自用 CUDA core 讀一遍 KV，樹驗證 12 個節點就讀 12 遍；現在一個 MMA 的 16 列放「query head × query」，
   5 個 query 以內一個 warp 一次讀 KV 給全部用。K/V 直接從顯存進 MMA 的暫存器（不經 shared memory），FP8 用兩個位元運算變 FP16。
   3060 Ti 上單獨量（200K、每個 visit）：單 token 2.17 → 1.19 ms（193 → 346 GB/s，峰值 448），樹驗證 12 節點 22.3 → 2.2 ms。
   跟原 kernel 的輸出差約 1e-3（FP16 捨入）；逐 token 解碼的 KLD（`INK_KLD_DECODE_FROM=1024`，雙卡 fp8）新 0.005109、舊 0.005436。
9. DSpark 草稿器的全域層注意力也改用同一個 kernel（FP16 KV 版）。原本每個 (query, head) 一個 block、每個 key 做一次 warp 加總，
   長度越長越慢：64K 時草稿每步 14.6 ms（2K 時 5.9 ms），單獨量 200K 每次 27.7 ms → 2.5 ms（主卡）。這才是長文投機輸給不投機的原因。
10. 讀 prompt 時兩張卡同時做（`ink_runner.cuh` 的 `emit_two_lanes`）：一段 2,048 token 拆成兩半，各由一條 CPU 執行緒送工作，
    輪到誰才送（同一條 stream）。前半在 3060 Ti 算全域注意力時，主卡算後半到它自己的全域 visit，再交換。每個 visit 都是前半先做，
    所以 SWA 環狀快取、ShortConv 狀態、第二張卡的 KV 順序都跟原本一樣。跨過全域 visit 還要用到的暫存（殘差、block、bank、taps 等）
    用同一塊記憶體的不同列，不多佔顯存。交接改成兩組請求／回覆緩衝區、請求編號、第二張卡用自己的計數回覆（`ink_remote_attn.cuh`）。
    KLD（雙卡、chunk 512）開 0.004413、關 0.004478；Top-1 96.05% vs 96.78%，跟不拆但 chunk 256 的 95.83% 同一級（切塊大小造成的差）。
11. 試過不採用：3060 Ti 的 QKᵀ 改 FP16 累加（GeForce 上 tensor core 快一倍），200K 只快 1.7%（126 vs 128 秒），KLD fp8 KV
    0.005471 → 0.005612，不值得。`--chunk 3072` 200K 讀 prompt 124.6 秒（2,048 是 129 秒），但主卡只剩 199 MB、峰值 15.45 GB，
    超過約 14.2 GB 的安全線，維持 2,048。4,096 直接爆顯存（解碼 2.3 tok/s）。

第四輪（主卡讀 prompt 的 kernel）：

12. 先量 tensor core 實際峰值（`mma.sync` 空跑）：5060 Ti INT8 175.6 TOPS、FP16 輸入 FP32 累加 42.0、FP16 累加 83.8 TFLOPS；
    3060 Ti 138.8／35.0／70.0。專家（S2_K／S3_K，INT8 tensor core）讀 prompt 時只跑到約 30 TOPS，是主卡最大的一塊
    （單卡 8K：gate/up 32%、down 16%）。
13. 讀 prompt 時（一次 1,024 個 token 以上）專家改成 128 列一塊：2,048 個 token 時每個專家平均分到約 128 列，
    原本 64 列一塊，同一份權重要解壓兩次。少數 SF4 專家的 kernel 最多 64 列，排序時多寫一份 64 列的清單給它用。
    專家時間 gate/up 1,138 → 812 ms、down 555 → 386 ms（單卡 8K，nsys 加總）。每個輸出的算法和順序不變，KLD 跟上一版完全相同。
14. 注意力前的混合 kernel（`k_ink_attn_res_3x_cat_rms`）一個 token 有 4 個 block 讀同一批資料，原本這 4 個 block 在排程上
    隔 2,048 個 block，資料早就不在 L2；改成相鄰：1.35 → 0.78 ms。專家輸出的 8 路相加改用 float4：0.456 → 0.421 ms。
    單卡 8K 讀 prompt 的 GPU 時間合計 3.56 → 2.89 秒。
15. 試過不採用：專家 k-loop 雙緩衝（暫存器變多，每個 SM 只放得下 1 個 block，反而慢）、把 INT8 轉 float 的 I2F 換成加常數的技巧
    （沒有差）、128 列改 2×4 的 warp 排法（慢）。`__launch_bounds__(256, 1)` 會讓編譯器把暫存器從 128 加到 160–200，
    每個 SM 少一半 block，要寫成 `(256)`。

量精度時注意：從 `build-inkling/` 直接跑的 exe 會載入 CUDA 13.4 的 cuBLAS，`engine-local/inkling/` 裡的用發行包附的；
同一份原始碼兩邊 KLD 就差 0.0038。要比，先把 exe 放到 `engine-local/inkling/`。

### 量到的（bench，合成 prompt 全部真的讀進去，fp8 KV）

| 設定 | 讀 200K prompt | 200K 填滿後解碼 | 4K 短文解碼（DSpark／不投機） |
|---|---|---|---|
| 原版引擎，單卡 200K（專家 24% 在主記憶體） | 678 秒 | 10.0 tok/s | 49.3 tok/s（1K） |
| 兩張卡，第一版（cuBLAS 注意力） | 364 秒（平均 549 tok/s） | 22.8 tok/s | 106.5／39.3 tok/s |
| 兩張卡，第二輪 | 185 秒（平均 1,101 tok/s） | 26.3 tok/s | 117／60 tok/s（單卡、2K prompt 量的） |
| 兩張卡，第三輪 | 129 秒（平均 1,550 tok/s） | 42.6 tok/s（投機）／36.0（不投機） | 85–89 tok/s（雙卡、2K、投機） |
| 兩張卡，現在（第四輪） | **118.7 秒**（平均 1,684 tok/s） | 40.3 tok/s（投機，128 個 token） | 73–75 tok/s（server 短答） |

第四輪同一天對照（雙卡、含看圖模組、200K、生成 128 個 token）：第三輪的 exe 125.4 秒、解碼 40.5 tok/s；新版 118.7 秒、40.3 tok/s。
讀 prompt 每 8K 的速度（兩張卡，第四輪，含看圖模組）：64K 2,242、131K 1,602、197K 1,087 tok/s（第三輪同一天 1,942、1,563、1,082）。
雙卡 64K prompt：33.0 → 31.4 秒（只改專家 128 列，兩輪交錯）。
長文解碼（雙卡，投機／不投機，tok/s）：32K 60.6／47.0、64K 53.3／42.3、128K 49.7／36.4、200K 42.6／36.0。
單卡 112K 設定：32K 65.8／44.6、64K 53.3／40.7、100K 48.2／43.2。投機時 200K 每步平均 2.03 個 token（合成 prompt）。
16K prompt（單卡）：1,408 → 2,207 tok/s。單卡 112K 設定：32K 讀 prompt 1,916 tok/s。

透過 server（真實文字，repo 的文件，中間藏一句暗號）：

| 設定 | 長文 | 讀完並回答 | 暗號 |
|---|---|---|---|
| 200K 雙卡（第一版） | 178,435 token | 320 秒 | 答對 |
| 200K 雙卡（第二輪） | 172,981 token | 152 秒 | 答對 |
| 200K 雙卡（第三輪） | 164,100 token | 104 秒 | 答對 |
| 200K 雙卡（第四輪） | 156,703 token | 85.8 秒 | 答對 |
| 200K 雙卡（現在） | 60,271 token | 42 秒 | 答對 |
| 112K 單卡（第二輪） | 101,986 token | 68 秒 | 答對 |
| 112K 單卡（第三輪） | 99,177 token | 72.5 秒 | 答對 |

第三輪 20 分鐘穩定性測試（雙卡、server，短文與 28K–143K 長文輪流）：21 輪 0 失敗、server log 0 個錯誤，
143K 長文每次 85–90 秒（第二輪 117 秒）。

短答（中文、英文程式題）：雙卡 54–76 tok/s、單卡 57–81 tok/s（第二輪，含讀 prompt；第三輪同一題 45–65，回答長度不同，
同條件的 bench 新版 85–89、舊版 81）。開投機後 64K 單卡和雙卡一樣（都 53.3 tok/s），雙卡只在超過 112K 時才需要。

### 離極限還差多少

- 200K 讀 prompt 118.7 秒。前段卡在主卡（64K 時 2,242 tok/s），後段卡在 3060 Ti 的全域注意力（197K 時 1,087 tok/s，
  兩輪都一樣）。全域注意力約 2 PFLOP，3060 Ti 實測 FP32 累加峰值 35 TFLOPS，滿載也要 56 秒，現在約 7 成。
  主卡剩下的大塊：專家（INT8 只用到峰值約 2 成，卡在 k-loop 的同步和延遲，沒有效能計數器權限可以細看）、
  BF16 投影矩陣（cuBLAS，已接近 FP32 累加的峰值 42 TFLOPS，要更快只能改 FP16 累加或量化，會影響精度）。
- 200K 解碼：不投機 36.0 tok/s（每個 token 約 28 ms，3060 Ti 的 KV 讀取約 10 ms，已到約 8 成頻寬），投機 42.6 tok/s。
  投機每步約 2 個 token，驗證一步（約 4.4 個節點）的主卡時間約是單 token 的 1.8 倍（每個節點選的專家不同，要讀更多權重）。
- 3060 Ti 只負責全域注意力。把專家也分過去，每層都要跨卡交接（每次約 0.13 ms，每個 token 48 次），解碼會更慢，所以沒做。
