# 設定

Barbet 使用 `BarbetConfig`，它是標準的 Transformers `PretrainedConfig`。

> 注意：Hugging Face 上目前發布的是 29 層、原生 1M 的 checkpoint。它的設定副本在 `configs/barbet_1b_base/config.json`，對應的工廠方法是 `BarbetConfig.barbet_1b_base()`。`barbet_1b` 與 `barbet_1b_1m` 是早期 28 層 R2 的研究設定，和目前發布的權重不相容。完整差異見[長前後文](long_context.md)。

## 預設設定

| 設定 | 上下文長度 | 滑動視窗 |
| --- | ---: | ---: |
| `barbet_1b_base`（發布版） | 1048576 | 無（22 層全域注意力 + 7 層 Mamba2） |
| `barbet_1b` | 262144 | 8192 |
| `barbet_1b_1m` | 1048576 | 8192 |

所有設定都使用固定的 `OpenFormosa/PangolinTokenizer` 詞彙表，並內建以下標準 token id，所以生成時的停止與補齊行為會自動和 tokenizer 一致：

- `unk_token_id=114688`（`<unk>`）
- `bos_token_id=114689`（`<s>`）
- `eos_token_id=114690`（`</s>`）
- `pad_token_id=114691`（`<pad>`）

## 檔案

- `configs/barbet_1b_base/config.json`（與 Hub 權重一起發布的設定，手動維護，不由指令重新產生）
- `configs/barbet_1b/config.json`
- `configs/barbet_1b_1m/config.json`

每份設定都包含以下欄位，這是 Hugging Face remote code 載入所必需的：

```json
"auto_map": {
  "AutoConfig": "configuration_barbet.BarbetConfig",
  "AutoModel": "modeling_barbet.BarbetModel",
  "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM"
}
```

## 重新產生設定檔

```bash
PYTHONPATH=src python scripts/write_model_configs.py
```

這個指令會用 Python 的工廠方法重新寫出設定資料夾：

- `BarbetConfig.barbet_1b()`
- `BarbetConfig.barbet_1b_1m_extension()`

## 載入指定的預設設定

```python
from barbet import BarbetConfig

release = BarbetConfig.barbet_1b_base()
config_1b = BarbetConfig.barbet_1b()
config_1m = BarbetConfig.barbet_1b_1m_extension()
```

`barbet_1b_1m_extension()` 是原始 R2 權重的推論時長上下文外推設定，與 256K legacy 權重相容，只調整 RoPE 縮放資訊與最大上下文長度。它不是目前 Hugging Face iter7008 的 native-1M release config。詳細說明請見 [長上下文](long_context.md)。
