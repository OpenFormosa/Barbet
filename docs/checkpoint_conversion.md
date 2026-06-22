# 檢查點轉換

本專案提供一支把 Megatron `torch_dist` 檢查點轉換成 Hugging Face `safetensors` 的工具，適用於 Barbet 1B。

## 指令

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/open_formosa_1b_checkpoint \
  --output-dir /path/to/barbet-1b-hf \
  --force
```

`--checkpoint` 可以指向包含 `latest_checkpointed_iteration.txt` 的檢查點上層資料夾，也可以直接指向某個 `iter_0000xxx` 資料夾。

## 輸出內容

轉換完成的 Hugging Face 檢查點應該包含：

```text
config.json
config_1m_extension.json
configuration_barbet.py
modeling_barbet.py
model.safetensors
conversion_report.json
```

## 載入轉換後的模型

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/converted-barbet",
    trust_remote_code=True,
)
```

轉換後的 `config.json` 是原生 256K 的設定，用來維持與原始檢查點一致的解碼行為；1M 外推設定則另外寫在 `config_1m_extension.json`，兩份設定使用同一份 1B 權重。
