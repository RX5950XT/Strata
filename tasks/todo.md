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
