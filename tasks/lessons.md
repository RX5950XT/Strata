# 可重用教訓

- 使用者問「CPU 算力還是 RAM 頻寬」時，先做可分離的實測：同一個實際 kernel 比較快取內／RAM 工作集，再控制 CPU 時脈並量純讀取速度。只量 host 總時間，或反覆列出可能原因，沒有完成這個問題。
- Strata 這台 5700X 的 PBO 從始至終已開啟（使用者確認）。不能把「開 PBO」當新改善；Windows 電源方案停用 boost 的對照不等於 BIOS PBO 開關測試。其他機器也先確認既有狀態，不能由 Balanced 模式或時脈猜 PBO 是否開啟。
- 硬體瓶頸不等於購買建議。提出升級前先查當前價格與完整換平台成本，優先保留既有主機板及大容量 RAM；記憶體漲價時不能只按規格推薦整套 AM5／DDR5。沒有跨硬體實測，就不能宣稱核心增加一定划算或保證 tok/s 增益。
- Colab CLI：`colab exec` 會等整個指令的 stdout 關閉。`a && b &` 背景跑的是整個子 shell，會握著管線讓 kernel 卡死，之後所有 exec 都排隊；要背景就用單一 `setsid nohup cmd > log 2>&1 < /dev/null &`。`pkill -f 模式` 會比對到自己這個 bash 的指令列而把自己殺掉，改用 `pgrep` 拿 PID 再 kill。設定 `LD_LIBRARY_PATH` 要接在原值前面，Colab 的 libcuda 在 `/usr/lib64-nvidia`。
- 使用者要 Colab 連 Google Drive 時，用一次授權就能長期重用的方式（rclone token 存本機，每台新 VM 帶過去），不要用每台 VM 都要重新同意的 `colab drivemount`；先講清楚為什麼還要點一次授權。
