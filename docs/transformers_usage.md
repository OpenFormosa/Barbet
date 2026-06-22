# Transformers 使用方式

## 本機 Python 套件

```bash
pip install -e ".[dev]"
```

```python
import torch
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_1b()
model = BarbetForCausalLM(config)

input_ids = torch.randint(0, config.vocab_size, (1, 16))
outputs = model(input_ids=input_ids)
print(outputs.logits.shape)
```

## 載入設定資料夾

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "configs/barbet_1b",
    trust_remote_code=True,
)
```

直接從本機設定資料夾載入時，如果你想用 `AutoModel` 建立模型，請確認資料夾裡也放了 remote code 檔案。

## Hugging Face Hub 檔案配置

一個模型庫至少應該包含：

```text
config.json
configuration_barbet.py
modeling_barbet.py
model.safetensors
```

如果只想發佈設定檔，可以省略 `model.safetensors`。

## 從 Hub 載入

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("openformosa/PangolinTokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "openformosa/barbet-1b-base",
    trust_remote_code=True,
)
```

如果只想檢視設定，或在 HF 權重還沒發佈之前：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
print(config.max_position_embeddings)
```

Hub 模型庫可能會把 1M 延伸設定當成預設的 `config.json`，並把原生的 256K 設定保留成 `config_256k.json`。這兩份設定使用同一份 1B 權重，差別只在 RoPE 縮放資訊與最大上下文長度。

## 文字生成

```python
prompt = "台灣的健保制度"
inputs = tok(prompt, return_tensors="pt")
ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(ids[0], skip_special_tokens=True))
```

`generate()` 預設使用逐步解碼（`use_cache=True`），會把先前算過的狀態快取起來，所以每生成一個新 token 只需要算一次單一 token 的前向傳遞，不必重新計算整段序列。傳入 `use_cache=False` 可以強制重新計算整段；兩種方式產生的 token 完全相同。

提供的設定檔已經內建 PangolinTokenizer 的標準 token id（`eos_token_id=114690`、`pad_token_id=114691`），所以生成時的停止條件與補齊（padding）行為會自動和 tokenizer 一致，不需要額外指定參數。

## 儲存模型

```python
model.save_pretrained("barbet-1b-local")
config.save_pretrained("barbet-1b-local")
```

如果之後要用 remote code 的 `AutoModel` 載入，請再把 `configuration_barbet.py` 與 `modeling_barbet.py` 複製到同一個資料夾。
