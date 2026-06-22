---
language:
- zh
- en
language_bcp47:
- zh-Hant
library_name: transformers
pipeline_tag: text-generation
tags:
- causal-lm
- custom-code
- long-context
- mamba
- open-formosa
- barbet
license: other
---

# Barbet 1B Base

`openformosa/barbet-1b-base` 是 Barbet 1B 基礎模型在 Hugging Face 上的封裝版本。Barbet 是一個 decoder-only 的混合式因果語言模型，預設使用 `openformosa/PangolinTokenizer` 詞彙表。

## 上下文長度

1B 基礎模型的目標上下文長度是 256K。另外提供一個推論時的 1M 外推設定，使用同一份 1B 權重：

```json
{
  "max_position_embeddings": 1048576,
  "rope_scaling": {
    "type": "linear",
    "factor": 4.0,
    "original_context_length": 262144
  }
}
```

這並不是原生的 1M 預訓練。實務上要跑到 1M 等級的長上下文，仍需要額外經過最佳化的長上下文執行環境。

## 載入方式

只檢視設定：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
print(config.max_position_embeddings)
```

載入權重：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openformosa/PangolinTokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "openformosa/barbet-1b-base",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

Hub 上的 `config.json` 是原生 256K 的設定；1M 外推設定保留成 `config_1m_extension.json`，兩份設定使用同一份 1B 權重。

若要得到最接近原始模型的解碼結果，請在 CUDA 上執行並安裝 `mamba_ssm`。沒有 `mamba_ssm` 時，模型會改用內建、可攜性較高的 PyTorch Mamba 路徑。

## 適用範圍

Barbet 1B Base 是一個基礎語言模型，適合用於正體中文、多語預訓練，以及長上下文檢索行為等研究。它不是經過指令微調的助理模型；若要做面向使用者的助理應用，請改用經過指令微調或安全對齊的版本。

## 使用限制

- 1M 設定是推論時的 RoPE 外推設定，不是原生的 1M 訓練。
- 只有 CPU 時，Mamba 會使用 PyTorch 後備路徑；在 CUDA 上搭配 `mamba_ssm` 才能得到最接近原始模型的解碼路徑。
- 基礎模型在沒有解碼限制或指令微調的情況下，生成內容可能會重複或偏離主題。
