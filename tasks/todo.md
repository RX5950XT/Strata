# 上游 0.2.19 合併＋Inkling-2-Mini 去拒答版（2026-11-22）

- [x] 合併 upstream/main（0.2.11 → 0.2.19，引擎 0.1.65，2,545 個 commit）。上游搬目錄：`src/`→`engine/src/`、`serve/`、`setup.py`、`data/`→`terminal/`。11/23 合併進 main 並 push，分支已刪。
- [x] fork 的資料仍在 repo 根目錄（configs/、models/、packs/、engine-local/、.panel/）；面板用 `REPO = terminal/..` 找它們，server 從 `terminal/` 跑。SC117 設定的 expert-profile 改指 `terminal/data/`。
- [x] 雙卡主引擎用 `build-dual.bat`（`-S engine`，vendored ggml）重編，engine-local 換 0.1.65，舊版 `engine-local/strata-0.1.57.exe`。SC117 雙卡 A/B 一輪：解碼新 42–48、舊 42–50 tok/s，worker 正常。
- [x] Inkling：跑官方 SQ-10＋OpenUncensored 的 20 個 `wo_ud`（`tools/inkling_uncensor_patch.py`），單卡、fp8 KV、看圖、DSpark。數字與原因在 docs/INKLING-UNCENSORED.md。
- [x] Inkling 長上下文（11/23）：讀 prompt 注意力分段（固定記憶體）、DSpark 環狀緩衝、`--kv-gpu 1` 把全域 KV／注意力放 3060 Ti（門鈴＋keep-warm）、64 段 split-K、`--draft-max-pos`。200K 兩張卡：prompt 364 秒、解碼 22.8 tok/s；112K 單卡設定照舊。數字與極限在 docs/INKLING-UNCENSORED.md。舊 exe：engine-local/inkling/strata-inkling-112k.exe、-ctx64k.exe。
- [x] 離極限第二輪（11/23）：融合 flash prefill（`ink_pf_flash.cuh`，FP16 MMA，KLD 比原版好）、詞表輸出層直接讀 S6_K（每 token 少搬約 2 GB）、16-lane 解碼注意力、`--chunk 2048/1024`、`--draft-max-pos 40960`。200K 雙卡：prompt 185 秒、解碼 26.3 tok/s；面板預設改成雙卡 200K（設定 `"default": true`），單卡改名 `-112k`。舊 exe：`strata-inkling-prev.exe`。
- [x] 第三輪（11/23）：解碼／樹驗證注意力和 DSpark 草稿全域層改 tensor core（`ink_dec_tc.cuh`）、讀 prompt 兩張卡同時做（一段拆兩半交錯，`emit_two_lanes`，交接改兩組緩衝區）。200K 雙卡：prompt 129 秒、解碼投機 42.6／不投機 36.0 tok/s；投機在每個長度都贏，兩份設定拿掉 `--draft-max-pos`。上一版 exe：`strata-inkling-r2.exe`。
- [x] 第四輪（11/23）：專家讀 prompt 改 128 列一塊（SF4 另一份 64 列清單）、注意力前混合 kernel 排程改相鄰、8 路相加 float4。KLD 不變；200K 雙卡 prompt 125.4 → 118.7 秒（同一天對照），156,703 token 真實長文 85.8 秒答對。上一版 exe：`strata-inkling-r3.exe`。
- [x] HIP：WSL ROCm 7.2.4（補裝 hipblas/rocblas/hipblaslt-dev）gfx1100 編得過、`--help` 能跑；連結要 `-DCMAKE_EXE_LINKER_FLAGS=-no-pie`（上游 HIP 物件沒 -fPIC，跟這次改動無關）。SYCL：WSL 裝 oneAPI 2026.2，`build_inkling_sycl.sh jit` 原始碼全編過，連結缺 `quant::dequant_row`（上游腳本少編一個檔，跟這次無關）；SYCL 引擎沒用到改過的共用檔。
- [x] 第五輪（11/23）：GPU 效能計數器登錄檔已開（`NVTweak\RmProfilingAdminOnly=0`，重開機後生效；之前用管理員跑 ncu）。讀 prompt 專家 activation 改 `cp.async` 預取（−3～−11%）；驗證專家 kernel 限制每 SM 3 個 block（255 → 168 暫存器，gate/up −16%、down −10%）。KLD 與輸出 token 完全相同（`INK_BENCH_TOKENS`）；雙卡 200K 117–121 秒，跟上一版一樣。HIP 編過，SYCL 沒用到這兩個檔。上一版 exe：`strata-inkling-r4.exe`。
- [x] 第六輪（11/24）：讀 prompt 專家把 S2_K／S3_K superblock 用 `cp.async` 整塊搬進 shared memory（不改檔案格式，−9%，單卡 8K 讀 prompt +5～10%、雙卡 64K −2～3%）；投機驗證 q8 投影每 block 16 列（−11%，整步約 −2%）。KLD 與輸出 token 完全相同。短文解碼已貼著這張卡讀 2–20 MB 的實際速度（只讀就 140–300 GB/s），「122 tok/s 上限」不成立。上一版 exe：`strata-inkling-r5.exe`。
- [x] 第七輪（11/24）：跨請求 prompt 快取（讀到倒數第 1 個 token 存非位置狀態到主機記憶體，下一輪開頭相同就還原；39K 對話之後每輪 22 → 4 秒，對齊切段的測試輸出完全相同）；讀長文注意力 split-K（3060 Ti kernel −12～−18%，192K 115 → 108 秒）；解碼 F32 小矩陣一列一個 block、滑動視窗層改 tensor core kernel（每步 −1 ms，約 +3%）。HIP gfx1100 編過；SYCL 原始碼編過（連結缺 `quant::dequant_row` 是上游舊問題）。上一版 exe：`strata-inkling-r6.exe`。
- [ ] 還沒做：3060 Ti 注意力已是 FA2 水準（峰值 7 成多），再快要 FP16 累加（精度變差）；解碼剩下約 1,400 個小 kernel／步，要整層合併。
- [x] Abiray Q3_K_M GGUF：載入器加主線命名、EP 後端加 Q3_K／Q5_0／Q5_1、修 `d_dense_x_` 溢位；能跑但只有 EP 慢路徑（約 5 tok/s）。
- [x] 面板支援 Inkling（DSpark、引擎內建看圖、從 server log 讀速度）；PANEL.bat 改 `terminal\serve\panel.py`。
- [x] setup 雙卡設定移植（11/23）：`config_for` 補回 `expert_setup`（主卡 uuid＋`expert_gpu`＋`--expert-gpu 1`，拿掉 `gpu_roles`）與看圖放第二張卡；`configs_dir()` 在這個 checkout 指到根目錄 `configs/`（setup 之前看不到已裝的模型、寫到 `terminal/`）。`move_over_old_layout` 遇到根目錄有 PANEL.bat 就不搬（它曾把 packs/、mtp/、engine/*.exe 搬進 terminal/，已搬回）。setup 測試 14 個檔由失敗轉通過，golden 補回 `--vram-reserve-mib 700`。
- [x] 穩定性（11/23，經面板 API，短文與 30K～150K 字藏碼長文交替）：Inkling 雙卡 200K 跑 23 輪 0 錯（143K token 長文 76 秒）；SC117 雙卡 20 分鐘 21 輪 0 錯（137K token 約 90 秒），log 無錯，兩張卡 VRAM 沒有往上爬。SC117 是會思考的模型，答案可能在 `reasoning_content`，`max_tokens` 要給夠。
- [ ] serve 測試 8 個在純上游一樣失敗（inkling_serve_e2e 2、server_script_mode、stop_closed_pipe 5）；未做 Metal 編譯。

# 上游 0.2.11 合併

- [x] 合併 upstream/main（0.2.6 → 0.2.11，引擎 0.1.57），10 檔衝突，保留雙卡 worker。分支 `merge/upstream-0.2.11`，未 push。
- [x] 修上游「看圖要用時才啟動」誤判第二張卡的看圖模組（主卡 cache 少 1000 格、prompt 區塊減半）。
- [x] 舊新交錯 A/B：雙卡解碼與讀 prompt 都沒變慢，數字在 docs/DUAL-GPU.md。
- [x] engine-local 換成 0.1.57，舊版留在 `engine-local/strata-0.1.51.exe`。
- [ ] HIP／SYCL／Metal 未編譯驗證；未做長時間穩定性測試。

# 上游 0.2.6 合併＋DeepSeek V4.1 去拒答版上 Colab G4

- [x] 合併上游 0.1.45 → 0.2.6（1,758 個 commit：桌面 App、`strata launch`、引擎改首次啟動下載、DeepSeek V4.1-Flash 獨立引擎、Qwen3.6、Gyro、Intel Arc、DFlash2）。13 檔衝突保留 fork 的雙卡 worker、orca、configs/ 路徑、NO_WINDOW、STRATA_HIP_SWAR_SIGNS。
- [x] 驗證：build-dual 編譯成功；setup 測試 menu／choices／golden 通過（golden 依上游「預設開圖片」重錄）；serve 1035 個測試 1 個錯（test_winjob，合併前在本機也錯）；SC117 雙卡 32K 一輪 A/B 解碼 38.6→43.5、29.3→34.1 tok/s，開頭輸出相同。engine-local 已換新，舊 exe 在該 session scratchpad old/。未 push。
- [x] Colab G4 部署 Abiray IQ2_XXS＋DSpark（MTP）＋視覺塔，context 204,800；MTP 開約 93 tok/s、關約 64；其他載入參數無差。細節：bench/results/2026-10-26-colab-g4-ds41/。
- [x] 整套 241 GiB 放 Google Drive `Strata-DS41/`；WSL `ds4-up` 一鍵開新 G4 從 Drive 拉回並啟動。
- [ ] `ds4-up` 尚未在新機器實跑；200K 請求未完成；HTTP server 尚未把圖片送給 DeepSeek 引擎。

# HIP IQ2/IQ3 正負號技巧（借鑑 llama.cpp）＋ 上游檢查

- [x] 上游 Niko1221/Strata 最新 477ac13c（10/16）已合併，無新更新。
- [x] iq_kernels.cu：32 處 `__vcmpne4`/`__vsub4` 改經 `sign_mask`/`apply_sign_mask`；CUDA 與 HIP 預設為巨集展開成原碼。
- [x] HIP 新寫法改為 opt-in（`-DSTRATA_HIP_SWAR_SIGNS=1`）：靜態加權 VALU −25%，但 17–19 個 kernel occupancy 下降，無 AMD 卡無法實測。
- [x] 驗證：CUDA SASS（sm_86/sm_120）與 HEAD 逐位元相同；HIP 預設 gfx1100/gfx1201 組語相同；公式 48,020 例 0 錯。
- [x] 修 tools/strata_bench.py 自行啟動 server 時 `port` 未定義（上游 #2330 引入）。
- [x] NVIDIA A/B 測速（tools/strata_bench.py，iq3_xxs dual config，old/new 交替 2 輪，各 3 runs 中位數）：decode 34.6/28.6 vs 31.1/31.5、4K prompt 1090/1068 vs 1077/1083、16K 1381/1370 vs 1370/1366 tok/s——落在雜訊內，符合 SASS 相同的預期。

Review：prefill `moe_fused_iq.cu` 的同類修改與 `v_perm` 廣播在 gfx1100/gfx1201 無效果，已撤回。WSL 已安裝 ROCm 7.2.4 編譯器（hipcc、rocm-llvm、hip-dev）供 HIP 編譯檢查。

# CPU／RAM 瓶頸實測

- [x] 使用實際 orca IQ3_XXS expert 與既有 CPU kernel，比較 CPU 快取／RAM 讀取。
- [x] 同時量純讀取速度、CPU 時脈估計及可用硬體 counters，交錯重複測量。
- [x] 以可還原的 CPU boost 控制量測時脈；較低時脈限制未實現，沒有冒稱成功。
- [x] 保存原始數據、還原環境，回報主要限制及驗證邊界。

完成標準：有同工作量的對照數據，可以判定主因；不能以 CPU 使用率或有效 GB/s 直接冒充 RAM 飽和量測。

Review：同一程序內兩批共 384 個短段（正常／停用 boost 各半），正常組每個 token 數 48 對；移除大部分 RAM 讀取可省 21–27%，gate/up 仍為最大階段，定位 CPU 側處理為主，RAM 供料次之。DMA 對照證實供料競爭會增加 RAM 組耗時。完整 100k／200k 生成回合因輸出／路由不同，不作純時脈因果證據。正式 config 不變，原電源模式已還原，本次私有測試程序已關閉；完整紀錄在 bench/results/2026-10-14-local-5700x-cpu-ram/。

更正：使用者確認 PBO 全程開啟。上述 boost 對照只變更 Windows 電源方案，沒有切換 BIOS PBO；開啟 PBO 不是新的改善，進一步調整 PBO 參數的增益未驗證。

# 解碼速度還能擠多少（0.2.11）

- [x] 量解碼時間分布：CPU 專家約 41%，等 GPU 約 36%，預測模型約 7%，調整約 5%。
- [x] 交錯測 8 種設定，淘汰 `--spec-min-p`、`STRATA_ROUTE_TAIL_SKIP=6`。
- [x] 關遊戲後補跑，再用交錯順序確認，加測 190K prompt。
- [x] 設定檔改 `--kv-grow`、`--vram-reserve-mib 1036`，實跑確認能啟動並回答。

Review：數字在 docs/DUAL-GPU.md「讓主卡別塞太滿」。開遊戲那幾批不算數。沒有重測單卡，也沒有長時間穩定性測試。
