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

## 跟上游 0.2.20 比（2026-11-24）

上游 0.2.20（Inkling 引擎 0.1.66）重寫了 Inkling 引擎（四層專家放置、KV 放主記憶體、雙卡分層、原生 GGUF），
跟 fork 的引擎在 19 個檔案衝突。同一台電腦、官方 SQ-10、`--bench`、fp8 KV，兩邊交錯量（token／秒）：

| | 上游 0.2.20 | fork |
|---|---|---|
| KLD（CPU SQ-20 參考，512 token） | 0.1195 | 0.1155 |
| 讀 prompt：1K／4K／32K | 1,831–1,913／1,568–1,754／1,289–1,304 | 2,088–2,417／2,312–2,315／2,083–2,086 |
| 解碼，DSpark 開：1K／4K／32K | 138–147／140／57.8 | 127–148／157／97.3 |
| 解碼，DSpark 關：1K／4K／32K | 49.9–50.0／47.7–49.5／44.7 | 70.7–70.8／70.0／62.6–62.7 |
| 200K 讀 prompt／解碼 | 351 秒／21.7（`--gpus 2 --layer-split auto`）；428 秒／2.6（單卡＋KV 放主記憶體） | 103.6 秒／47.0（`--kv-gpu 1`） |

所以合併時 `engine/src/inkling2/` 和 `engine/include/strata/inkling2/` 整個留 fork 版（上游新增的 Inkling 檔案也拿掉），
其他部分照上游合併。合併前後同一個 exe 設定的輸出 token 雜湊相同（`33a179df6d9013b5`，4K、生成 512）。
上游 0.2.20 的 Inkling 修正裡，跟這台電腦有關的 #6617 已用同樣做法修；#6620（工具）、#6627（多輪前綴）在 server，
跟著合併進來，實測正常。上游引擎留在 `engine-local/inkling-up/`。

SQ-15、SQ-20 在 5060 Ti 單卡（去拒答版、`--max-context 65536`，交錯各兩輪）：專家放不完（SQ-15 287–418 個、
SQ-20 1,089–1,176 個在主記憶體），兩輪差很多。DSpark 關：1K 兩邊差不多，4K–32K fork 讀 prompt 快到約 2.4 倍
（SQ-15 32K 1,222–1,272 對 539–680；SQ-20 32K 735–805 對 338–361），32K 解碼 fork 也快（SQ-15 8.9–9.0 對 5.2；
SQ-20 13.2–15.0 對 6.4–7.4）。DSpark 開：fork 同一格兩輪最多差 7 倍，讀 prompt 比關掉慢（SQ-15 32K 189–791），
有主記憶體專家時 fork 的 DSpark 路徑還有問題沒查；上游開 DSpark 在 1K–4K 比較快也比較穩（SQ-20 1K 解碼
15.8–17.1 對 fork 5.8–8.8）。完整表格在 HF 模型卡。

## 已知問題

- Inkling 沒有關得掉的思考：模板送「Thinking effort level: 0」，中文題仍常先想一段，server 把思考段丟掉。`max_tokens` 太小會只剩空白或碎片，給 1,000 以上。
- 上游的雙卡 EP 後端（`--gpus 2`）比單卡慢約 50 倍，不要用；fork 的 `--kv-gpu 1` 是另一種分法（見下面 200K 一節）。

## 重做一次

```sh
python -I tools/inkling_uncensor_patch.py models/inkling2-strata/Inkling-2-Mini-SQ-10.strata models/inkling2-strata/Inkling-2-Mini-Uncensored-SQ-10.strata
```

工具先確認第 0–3 層（沒改過）重新編碼後跟官方位元組完全相同，才寫檔；寫完更新 Section 1 的 SHA-256（`--selfcheck --verify-hash` 通過）。

- 來源：官方檔 `strata-engine/Inkling-2-Mini-Strata` 的 `Inkling-2-Mini-SQ-10.strata`（commit `04996a1e`，SHA-256 `ac4d452a…5934`）；
  去拒答權重固定抓 `OpenUncensored/Inkling-2-Mini-Uncensored` 的 commit `4e5f06d7`（只讀約 340 MB）。
- 做出來的檔案 SHA-256 應為 `82cbd007fc7f08f7de33ae7912d36112af97b148effc2a9ed17306a5afa48a5e`。
  2026-11-24 重做一次，跟正在用的檔案逐位元相同。
- 工具也能做 SQ-15、SQ-20（BF16 的 `wo_ud` 直接複製）。三種大小已公開在
  [RX5950XT/Inkling-2-Mini-Uncensored-Strata](https://huggingface.co/RX5950XT/Inkling-2-Mini-Uncensored-Strata)；
  SHA-256：SQ-15 `93b244da…0d0d`、SQ-20 `6453a871…b671`。跟官方檔逐位元比，差異只在 20 個 `wo_ud` 和檔頭雜湊。
- 專家放不進顯示卡時（SQ-20 在 16 GB 卡上一定會），讀 prompt 要靠兩塊暫存區把主記憶體的專家搬上來算。
  0.1.65 在放完專家、配好固定主記憶體之後才配暫存區，常常沒配到（啟動紀錄 `pf_buf=0MB`；上游的說明是
  固定主記憶體的分頁表吃掉了顯示卡記憶體，這點我沒量）。這時每段 prompt 都悄悄跳過所有主記憶體的專家：SQ-20 回答多國語言混雜，
  KLD（以 CPU 跑的 SQ-20 為參考，512 token）1.663；模擬同樣情況、專家放更多到主記憶體時是 4.5。
  這是上游 #6617，0.2.20 已修（commit `4aaa1cbdc`）；fork 照同樣做法改：放完專家就從預留的 256 MiB 配暫存區，
  萬一沒有就報錯停下。改後同樣的模擬 `pf_buf=64MB`、KLD 0.015；SQ-10 有 245 個專家在主記憶體時，
  KLD 跟全部在顯示卡時完全相同（0.115477）。
  先前以為是 `encode_ue8m0` 的 `0xFF`，那是錯的（改了沒差，已撤回）。

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
   `INK_PF_PROF`（讀 prompt 每一段的時間，要配 `INK_LANES_OFF`）、`INK_MOE_TR128_OFF`（專家改回 64 列一塊）、
   `INK_BENCH_TOKENS=1`（bench 印出生成 token 的雜湊，比兩個 exe 的輸出是否一樣）。

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

第五輪（打開效能計數器之後，用 Nsight Compute 看 kernel 內部）：

16. 讀 prompt 的專家 kernel（`k_ink_moe_sk_tc`）：每個 warp 大部分時間在等資料（long scoreboard）和等同步（barrier），
    每個執行緒 128 個暫存器、每個 SM 2 個 block。權重每塊 70／102 位元組，沒對齊 16，一個位元組一個位元組讀，多讀了 71% 的區段。
    改成用 `cp.async` 在算這一組時先把下一組 activation 搬進 shared memory（兩份輪流用，d 和 s 跟著放在同一列的 64、68 位元組）。
    專家時間 gate/up 約 −3～−11%、down 約 −8～−11%（nsys，兩輪）；KLD 跟上一版完全相同（主卡 0.002115），
    單獨跑 3060 Ti（主機記憶體專家那條路）也一樣（0.002150）。權重不能用 `cp.async`（沒對齊）。
17. 投機驗證的專家 kernel（`k_ink_moe_gu_sk<4>`／`k_ink_moe_dn_sk<4>`，2～32 個 token）：gate/up 只用到顯存頻寬的 63–69%、
    down 37–42%，每個執行緒 255 個暫存器（7 種量化格式的解壓全部展開在同一個 kernel），每個 SM 只放得下 2 個 block。
    限制成每個 SM 3 個 block（168 個暫存器）：gate/up 161 → 134 us、down 90 → 80 us（單卡 2K、生成 256 個 token，nsys 三次）；
    限制成 4 個 block（128 個）會大量溢出，gate/up 反而慢 10%。只處理 1 個 token 的版本不動。輸出 token 完全相同
    （`INK_BENCH_TOKENS`，2K 和 8K）。端對端解碼的雜訊（同一個 exe 118–171 tok/s）比這個差別大，量不出來。
18. 雙卡 200K 讀 prompt（含 16、17）：新版 117.0／120.6 秒、上一版 117.6／118.8 秒，一樣。後段卡在 3060 Ti，主卡快一點看不出來。

第六輪（11/24，主卡的讀 prompt 專家和短文解碼）：

19. 讀 prompt 的專家 kernel：S2_K／S3_K 的 superblock（70／102 位元組）每次只用 1/4，原本 4 次各自從顯存讀，
    同一個 32 位元組區段讀好幾次。改成一列換到新的 superblock 時，用 `cp.async` 把包住它的 16 位元組對齊區段整塊搬進
    shared memory（最後一段只搬到 superblock 結尾，不會讀出界），4 次都從 shared memory 解壓。不用改檔案格式。
    專家時間（單卡 8K，nsys，兩輪）1,558 → 1,418 ms（−9%）；單卡 8K 讀 prompt 2,279–2,355 → 2,461–2,508 tok/s；
    雙卡 64K 29.4–29.7 → 28.6–29.1 秒。KLD 跟上一版完全相同（主卡 0.004628、3060 Ti 0.005332，refcpu），輸出 token 也相同。
20. 短文解碼離極限多遠：先量這張卡「只讀不算」的速度（同一時脈）：2.4 MB 140 GB/s、7 MB 210–243、20 MB 約 300、
    268 MB 也只有約 305 GB/s（Windows 下 GPU 時脈常在 1.9 GHz，不是 3.1 GHz）。不投機解碼的 q8 投影矩陣
    （每次 2–20 MB）7 MB 33 µs、20 MB 72 µs，已貼著這條線；一個 token 約 1,100 個 kernel，GPU 忙碌 93%。
    原本用 374 GB/s 算出的「122 tok/s 上限」不成立。
21. 投機驗證的 q8 投影（8 個 token 一起算）比只讀慢：每個 block 8 列時，每次要先把 8 個 token 的輸入（約 23 KB）
    搬進 shared memory，比要讀的權重還多。改成每個 block 16 列（256 執行緒）：這個 kernel 佔其他 kernel 的比例
    0.224–0.229 → 0.200–0.205（nsys，三次交錯），約 −11%，整步約 −2%。32 列沒有更快。輸出 token 相同。
22. 試過不採用：q8 投影先把權重讀進暫存器再搬輸入（沒差）；詞表輸出（S6_K，8 個 token）一個 warp 算 4 列共用輸入
    （暫存器變多、能同時跑的 warp 變少，2.4 → 3.5 ms）、2 列（±5%，沒差）。

第七輪（11/24）：

23. 多輪對話不再重讀前面的內容。原本每個請求都從頭讀整段 prompt（39K token 的對話每輪約 22 秒，同一題再問一次也一樣）：
    引擎只在「上一輪的 prompt 加上它自己生成的字」原封不動出現時才接著用，聊天程式送回來的格式永遠對不上；
    又因為滑動視窗層和短卷積只留最後的狀態，不能只用前半段。現在 prompt 有 1,024 token 以上（`INK_PROMPT_CACHE_MIN`，
    0 = 關）時，讀到倒數第 1 個 token 先停下，把和位置無關的狀態（40 個滑動視窗層的環狀 KV、`--kv-gpu` 的中轉環、
    短卷積、DSpark 的兩個視窗層和 `f_rnn`，200K 設定約 300 MB）存到主機記憶體；下一個開頭相同的請求還原它，只讀新的部分。
    全域 KV 照位置存放，本來就還在。只有 CUDA 後端；其他後端照舊。API 的 `cached_tokens` 也會顯示重用的數量。
    - 精確度：用引擎協定直接送 token，讓存檔點剛好在 2,048 的倍數，兩邊切段相同：接著算和從頭算的 200 個 token
      完全相同（雙卡多 500、3,000 token；單卡多 1,500 token）。切段位置不同時答案可能不同（數值的加總順序）。
    - 速度（server，雙卡）：39K token 對話第 1 輪 23 秒、之後每輪 4–4.5 秒（多半是生成）；97K token 第 1 輪 52 秒、
      之後 8–18 秒，從頭算 64 秒。
24. 讀長文的注意力（3060 Ti）：一次 1,024 個 query 是 128 個 block，38 個 SM 跑 3.4 輪，最後一輪大半空著。
    全域層 key 夠長時拆成最多 4 段（挑最後一輪最滿的段數）各自算，再用 `k_ink_pf_flash_merge` 合併（`INK_PF_SPLIT`）。
    單獨量 kernel（3060 Ti、fp8 KV）：16K–188K 從 19.4–22.3 → 23.6–25.4 TFLOPS（−12～−18%），結果和不拆差 2e-8；
    fp16 KV 也只多 6%，fp8 解壓不是瓶頸，剩下的已是 FlashAttention-2 在這代卡的水準（峰值 35 TFLOPS 的 7 成多）。
    雙卡 192K：115.0 → 107.6 秒，最後 8K 1,135 → 1,273 tok/s。
25. 解碼的小 kernel：投機一步約 27 ms、1,440 個 kernel。24–40 列的 F32 小矩陣原本只有 3–5 個 block（一個 warp 走完一列），
    改成一列一個 block：每步 0.71 → 0.19 ms。40 個滑動視窗層原本用舊的 split-K（2 MB 的 KV 要 39 µs），
    `k_ink_flash_dec_tc` 改成也能讀視窗（從第一個 query 往前 1,023 個、環狀位置、視窗遮罩，16 段）：注意力每步
    2.11 → 1.58 ms（`INK_DEC_TC_SWA_OFF=1` 回舊路）。解碼路徑 KLD（fp8，主卡）0.005264 → 0.005155；
    同一個 exe 只改切段大小 KLD 就差 0.0005，所以 F32 那項 0.004686 → 0.004814 在雜訊內。
    雙卡 4K、生成 512 個 token、交錯三次：122.1–122.4 → 123.4–126.9 tok/s。
26. 解碼剩下的空檔與小 kernel（11/24，試過、沒採用）：
    - 主卡單卡、1K prompt、投機一步約 40 ms（bench 的合成 prompt 每步約 7 個 token）。nsys 量到 GPU 閒著約 2.2 ms：
      0.6 ms 是 graph 內 kernel 之間的小空隙，1.6 ms 是等 CPU 讀結果再送下一批。
    - 試做「一步只等 CPU 一次」：樹驗證後由 GPU 自己走樹、提交、把接受的 token 餵進 DSpark（依接受數挑 8 種 graph 之一，
      跟原本同樣的 kernel 和列數）、接著先算下一步的草稿。輸出 token 跟原版完全相同，nsys 下閒置 1.6 → 1.1 ms；
      但不開 nsys 交錯量 6 次（各生成 2,048 個 token）兩版都是 190.0 tok/s，雙卡 16K 也一樣（18.4／18.3）。
      nsys 本身讓每次 API 呼叫變慢，把這段空檔放大了。沒有變快，所以沒有留。
    - 量一個很小的 kernel 在 graph 裡的成本：240 個串起來多 0.22 ms，一個約 0.9 µs（含它自己的工作）。
      每步不到 5 µs 的 kernel 約 780 個，就算全部消失也只省約 0.7 ms（每步的 1.7%）；真的合併只能省一部分開銷，
      量不出來（交錯量測的誤差約 ±1%），所以「整層合成一個 kernel」不做。

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
11/24 第八輪後重量（雙卡預設設定、bench、生成 256 個 token，各一次）：讀 prompt 1K 1,943、4K 2,575、16K 2,638、32K 2,563、
64K 2,564、128K 2,343、200K 1,911 tok/s（200K 共 104.7 秒）；不投機解碼 56.4、63.0、60.6、57.7、52.6、46.5、40.9 tok/s；
投機 75.0、88.1、55.2、106.0、60.4、111.6、50.7 tok/s（每步 1.5–4.5 個 token，跟合成 prompt 的內容有關，起伏比長度大）。
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

- 200K 讀 prompt 118.7 秒（第七輪 192K 107.6 秒，見第 24 點）。前段卡在主卡（64K 時 2,242 tok/s），後段卡在 3060 Ti 的全域注意力（197K 時 1,087 tok/s，
  兩輪都一樣）。全域注意力約 2 PFLOP，3060 Ti 實測 FP32 累加峰值 35 TFLOPS，滿載也要 56 秒，現在約 7 成。
  主卡剩下的大塊：專家（INT8 只用到峰值約 2 成，卡在 k-loop 的同步和讀權重的延遲，見第 16、19 點）、
  BF16 投影矩陣（cuBLAS，已接近 FP32 累加的峰值 42 TFLOPS，要更快只能改 FP16 累加或量化，會影響精度）。
- 200K 解碼：不投機 36.0 tok/s（每個 token 約 28 ms，3060 Ti 的 KV 讀取約 10 ms，已到約 8 成頻寬），投機 42.6 tok/s。
  投機每步約 2 個 token，驗證一步（約 4.4 個節點）的主卡時間約是單 token 的 1.8 倍（每個節點選的專家不同，要讀更多權重）。
- 投機驗證的專家 kernel 佔單卡投機解碼 GPU 時間約 25%（第 17 點之後），gate/up 讀顯存約 7 成頻寬。
  同一個專家被超過 4 個節點選到時，權重會每 4 個 token 重讀一次（多半從 L1/L2 拿到）。
- 短文解碼：大矩陣已貼著這張卡讀 2–20 MB 的實際速度（第 20 點）。小 kernel 和 CPU 往返加起來最多只佔每步約 2%，
  試過一步只等 CPU 一次也沒變快（第 26 點）；時間主要在專家（gate/up、down）和 q8 投影這幾個大 kernel。
  投機解碼的專家驗證在 24 個不同專家時，用平均專家大小估算也已到約 305 GB/s（估算，沒逐個專家量）。
- 3060 Ti 只負責全域注意力。把專家也分過去，每層都要跨卡交接（每次約 0.13 ms，每個 token 48 次），解碼會更慢，所以沒做。
