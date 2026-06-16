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

`voidful/barbet-1b-base` is the Hugging Face packaging target for the Barbet
1B R2 base model from the Open Formosa training stack. Barbet is a hybrid
decoder-only causal LM with global attention, sliding-window attention, and
Mamba-style sequence mixer layers.

## Current Hub Status

This initial publication contains the model card, config, and Transformers
remote-code files. HF-compatible `safetensors` weights are pending conversion
from the internal Megatron `torch_dist` checkpoint format. If no
`model.safetensors` file is present in this repository, use `AutoConfig` for
inspection only; `AutoModelForCausalLM.from_pretrained` will require converted
weights.

## Architecture

| Field | Value |
| --- | ---: |
| Model family | Barbet / Taiwan-Omni-1B-R2 |
| Layers | 28 |
| Hidden size | 1536 |
| FFN size | 5120 |
| Attention heads | 16 |
| KV heads | 2 |
| Head dim | 128 |
| Vocab size | 114944 |
| Tokenizer | `voidful/PangolinTokenizer` |
| Embedding / LM head | tied |
| RoPE theta | 10000000 |
| Sliding window | 8192 |
| Global attention layers | 0, 4, 8, 12, 16, 20, 24 |
| Mamba-style layers | 3, 7, 11, 15, 19, 23, 27 |
| QK logit clipping | disabled |
| Attention sink | disabled |

The layer motif repeats every four layers:

```text
global attention -> sliding attention -> sliding attention -> mamba-style mixer
```

## Context Length

The training target for the 1B base is 256K context. The 1M config is an
inference-time extrapolation config for the same 1B weights:

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

This is not a claim of native 1M pretraining. Practical 1M inference also needs
an optimized long-context runtime; the bundled PyTorch reference path can
express the RoPE scaling but global attention prefill remains quadratic.

## Loading

Config inspection:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("voidful/barbet-1b-base", trust_remote_code=True)
print(config.max_position_embeddings)
```

Once converted weights are present:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("voidful/PangolinTokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "voidful/barbet-1b-base",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

The Hub `config.json` may point at the 1M extension. The native 256K config is
kept as `config_256k.json`; both configs use the same 1B R2 weight shapes.

## Internal Evaluation Snapshot

These numbers are internal checkpoint-evaluation snapshots, not final public
benchmark claims. For cross-tokenizer comparisons such as MiniCPM, raw token
PPL is not comparable; use byte-normalized metrics such as bits per byte.

| Metric | Barbet 1B snapshot | Notes |
| --- | ---: | --- |
| TAIDE normalized LM loss | 1.066 bits/byte | tokenizer-normalized |
| Probability probes | 387 / 500 | internal QA-style probability probe |
| NIAH 32K | 32 / 32 | native context |
| NIAH 64K | 28 / 32 | native context |
| NIAH 128K | 25 / 32 | native context |
| NIAH 256K | 20-23 / 32 | native context, varies by run |
| NIAH 512K | 24 / 32 | extrapolated evaluation |
| NIAH 1M | 20-21 / 32 | extrapolated evaluation |

Needle-in-a-haystack mostly measures exact retrieval. It does not prove robust
multi-hop reasoning or full-context understanding at 1M length.

## Intended Use

Barbet 1B Base is a base language model for research on Traditional Chinese,
multilingual pretraining, and long-context retrieval behavior. It is not an
instruction-tuned assistant. Use an instruction-tuned or safety-aligned variant
for user-facing assistant applications.

## Limitations

- HF-compatible weights are pending conversion from Megatron distributed
  checkpoints.
- The 1M setting is an inference-time RoPE extrapolation config, not native 1M
  training.
- The reference implementation uses a self-contained PyTorch Mamba-style mixer;
  production kernels require a separate runtime integration.
- Base-model generation may repeat or drift without decoding constraints or
  instruction tuning.
- Raw token-level PPL should not be compared across tokenizers.
