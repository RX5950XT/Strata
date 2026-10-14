# CPU／RAM 瓶頸實測

- [x] 使用實際 orca IQ3_XXS expert 與既有 CPU kernel，比較 CPU 快取／RAM 讀取。
- [x] 同時量純讀取速度、CPU 時脈估計及可用硬體 counters，交錯重複測量。
- [x] 以可還原的 CPU boost 控制量測時脈；較低時脈限制未實現，沒有冒稱成功。
- [x] 保存原始數據、還原環境，回報主要限制及驗證邊界。

完成標準：有同工作量的對照數據，可以判定主因；不能以 CPU 使用率或有效 GB/s 直接冒充 RAM 飽和量測。

Review：同一程序內兩批共 384 個短段（正常／停用 boost 各半），正常組每個 token 數 48 對；移除大部分 RAM 讀取可省 21–27%，gate/up 仍為最大階段，定位 CPU 側處理為主，RAM 供料次之。DMA 對照證實供料競爭會增加 RAM 組耗時。完整 100k／200k 生成回合因輸出／路由不同，不作純時脈因果證據。正式 config 不變，原電源模式已還原，本次私有測試程序已關閉；完整紀錄在 bench/results/2026-10-14-local-5700x-cpu-ram/。

更正：使用者確認 PBO 全程開啟。上述 boost 對照只變更 Windows 電源方案，沒有切換 BIOS PBO；開啟 PBO 不是新的改善，進一步調整 PBO 參數的增益未驗證。
