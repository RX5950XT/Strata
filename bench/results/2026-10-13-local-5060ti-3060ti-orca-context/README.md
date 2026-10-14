# 本機 orca 長上下文速度

2026-10-13，Strata 0.1.43，本機 Windows 11。模型 `qwen3.8-flash-next-uncensored-iq3_xxs`，RTX 5060 Ti 16 GB + RTX 3060 Ti 8 GB，Ryzen 7 5700X；沿用 `configs/strata-orca-iq3_xxs.json` 的雙 GPU、vision、MTP、int8 KV 與 204800 context 設定。

| 目標輸入 | 實際輸入 | 輸出 | 預填 tok/s | 解碼 tok/s | TTFT 秒 | 總秒數 | MTP 接受率 | RAM 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 100,013 | 400 | 992.9 | 26.3 | 100.947 | 116.107 | 0.740 | 96,041 |
| 200,000 | 200,014 | 400 | 977.3 | 24.8 | 204.982 | 221.050 | 0.707 | 96,220 |

各測一次，前面先做 60-input / 30-output 的暖機；兩筆都是全新 nonce，engine log 顯示快取重用 0。速度取 engine 自己記下的 prompt/decode 計時，TTFT 與總時間取串流用戶端計時。提示詞用模型 tokenizer 計數，輸入提示包含 chat template 後實際多 13–14 tokens。

這次 200k 接在 100k 之後測，GPU expert cache 已較暖；結果不是多輪中位數，也不適合據此判定兩種長度的解碼差異。系統 RAM 峰值約 94.0 GiB（96,220 MiB）；Windows commit 最低餘裕約 1 GiB，兩次請求完成且 `/health` 仍為 `ok`，但當時幾乎沒有記憶體餘裕。不要把這組數字當成其他機器或不同快取暖度的保證。

- [results.md](results.md)：精簡數字
- [results.json](results.json)：完整設定、client metrics、engine metrics 與 GPU/RAM 取樣
- [engine-timing-lines.txt](engine-timing-lines.txt)：引擎 arena、請求速度與 commit log
- [host-timing/README.md](host-timing/README.md)：2026-10-14 解碼瓶頸分析
- [tuning-after-pagefile/README.md](tuning-after-pagefile/README.md)：CPU 優先權交錯測試、PCIe 比例與分頁檔調整
- [optimized/README.md](optimized/README.md)：保留設定後重新測量，100k／200k 解碼 35.9／31.9 tok/s
- [後續 MTP／CPU／cache 校準](../2026-10-14-local-5060ti-3060ti-orca-tuning/README.md)：沒有可穩定保留的新增益；還原原設定後，100k／200k 解碼重測為 36.8／34.4 tok/s
