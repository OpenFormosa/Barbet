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

# Barbet 1B Base — Stable Native 1M

`OpenFormosa/barbet-1b-base` 是一個約 1.119B parameters 的 decoder-only 混合式因果基礎語言模型，預設使用 `OpenFormosa/PangolinTokenizer`。目前發布的 checkpoint 是 `final-global-barbet-iter7008-stable-1m-v11`，原生 context length 為 1,048,576 tokens。

這不是 instruction-tuned assistant。模型沒有 chat template、SFT 或 RLHF；本次 long-context continued pretraining 使用一般 all-token causal next-token cross-entropy。

## 上下文長度

目前 release config 為：

```json
{
  "max_position_embeddings": 1048576,
  "rope_theta": 10000000.0,
  "rope_scaling": null
}
```

它不是舊版 256K checkpoint 的推論時 RoPE 外推，而是實際經過 exact-1M continued pretraining 的原生 1M checkpoint。完整訓練路徑見[〈Barbet 1B Base：我們如何把 256K 外推研究模型變成原生 stable 1M〉](../../docs/barbet_1b_native_1m.md)。

## Release 結果

- Fresh-loaded checkpoint 完成 140/140 個 exact 1,048,576-token evaluation rows，8/8 shards 全部完成，沒有 OOM、NaN 或 non-finite score。
- 七類 frozen long-context tasks 有 6/7 的 paired-bootstrap CI95 lower bound 大於零。
- Frozen base-model BPB retention 為 6/6 buckets，所有 bucket 均未比原始 Role-B 變差。
- Exact NIAH、opaque NIAH、multi-key、ordering、variable tracking、three-hop chain 通過；aggregation 尚未證明。

`stable 1M` 是有界的 base-model capability designation，不代表任意 1M 文件理解、可靠 free generation 或 universal aggregation。

## 載入方式

只檢視設定：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("OpenFormosa/barbet-1b-base", trust_remote_code=True)
print(config.max_position_embeddings)
```

載入權重：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("OpenFormosa/PangolinTokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "OpenFormosa/barbet-1b-base",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

請直接使用 Hub 上與 iter7008 權重一起發布的 `config.json`。GitHub repository 內的 `configs/barbet_1b/` 與 `configs/barbet_1b_1m/` 是原始 R2／legacy extrapolation presets，不是目前 release config。

若要得到最接近原始模型的解碼結果，請在 CUDA 上執行並安裝 `mamba_ssm`。沒有 `mamba_ssm` 時，模型會改用內建、可攜性較高的 PyTorch Mamba 路徑。

## 適用範圍

Barbet 1B Base 是一個基礎語言模型，適合用於正體中文、多語預訓練，以及長上下文檢索行為等研究。它不是經過指令微調的助理模型；若要做面向使用者的助理應用，請改用經過指令微調或安全對齊的版本。

## 使用限制

- Stable 1M 是 scoped likelihood-based capability claim，不代表完整理解任意一百萬 token 文件。
- Aggregation 尚未通過 exact-1M task-level statistical gate。
- 真正執行 1M 需要足夠 GPU memory、context parallelism 與相容的最佳化 kernels；單張消費級 GPU 通常不可行。
- 只有 CPU 時，Mamba 會使用 PyTorch 後備路徑；在 CUDA 上搭配 `mamba_ssm` 才能得到最接近原始模型的解碼路徑。
- 基礎模型在沒有解碼限制或指令微調的情況下，生成內容可能會重複或偏離主題。
