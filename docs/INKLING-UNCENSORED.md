# Inkling-2-Mini 去拒答版：這台電腦怎麼跑

量測日期 2026-11-22。機器：RTX 5060 Ti 16 GB（主卡）＋ RTX 3060 Ti 8 GB、Ryzen 7 5700X、96 GB RAM，Windows 11，引擎 strata-inkling 0.1.65（本 fork 自編，sm_86＋sm_120，CUDA 13.4）。

## 結論

- 跑的是 `models/inkling2-strata/Inkling-2-Mini-Uncensored-SQ-10.strata`：Strata 官方 SQ-10（專家約 2–3 bit），
  只把第 4–23 層的 attention 輸出矩陣 `wo_ud` 換成 OpenUncensored 去拒答版的權重。Abiray 的 GGUF 就是從這份去拒答版量化的；
  去拒答只改了這 20 個矩陣，專家、看圖、DSpark 草稿器都跟原版一模一樣。
- 只用主卡，上下文 64K，KV 用 fp8，看圖與 DSpark 投機解碼都開。
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

上下文（bench 只填 1K token，各跑兩次）：fp16／fp8 在 64K 以內 4 次都正常（約 1,600 tok/s、55–57 tok/s）；80K 以上時快時慢，
Windows 驅動把顯存不夠的部分挪到系統記憶體。真的把長文填進去才準：

- fp16、64K：58K 長文讀到 30,208 token 停住（顯存 15.8 GB 滿）。
- fp8、64K：58,146 token 讀完、正確找出藏在中間的暗號；讀 prompt 越後面越慢（前 15K 約 1,200 tok/s，53K–57K 約 49 tok/s，共 583 秒）。

## 已知問題

- Inkling 沒有關得掉的思考：模板送「Thinking effort level: 0」，中文題仍常先想一段，server 把思考段丟掉。`max_tokens` 太小會只剩空白或碎片，給 1,000 以上。
- 雙卡沒有好處：Inkling 引擎的雙卡只有 EP 後端，比單卡慢約 50 倍。

## 重做一次

```sh
python -I tools/inkling_uncensor_patch.py models/inkling2-strata/Inkling-2-Mini-SQ-10.strata models/inkling2-strata/Inkling-2-Mini-Uncensored-SQ-10.strata
```

工具先確認第 0–3 層（沒改過）重新編碼後跟官方位元組完全相同，才寫檔；寫完更新 Section 1 的 SHA-256（`--selfcheck --verify-hash` 通過）。
