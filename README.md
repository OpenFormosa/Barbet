# Barbet

Barbet is a Hugging Face Transformers implementation of the Barbet causal
language model family. The repository provides remote-code compatible modeling
classes and three configuration presets: Barbet 300M, Barbet 1B, and a Barbet
1B 1M research-extension config. The architecture mirrors the R2 revision of the
[Open Formosa](https://github.com/voidful/open_formosa) training stack
(Taiwan-Omni-300M-R2 / Taiwan-Omni-1B-R2).

This repository is intentionally lightweight. It contains model code, config
metadata, and checkpoint-conversion tooling. Megatron runtime artifacts remain
in the Open Formosa training stack.

## Contents

- `BarbetConfig`
- `BarbetModel`
- `BarbetForCausalLM`
- `configs/barbet_300m/config.json`
- `configs/barbet_1b/config.json`
- `configs/barbet_1b_1m/config.json`
- remote-code files for Hugging Face Hub loading:
  - `configuration_barbet.py`
  - `modeling_barbet.py`

## Model Summary

Barbet is a decoder-only hybrid language model with:

- grouped-query attention
- QK RMSNorm
- RoPE with large-context theta
- a repeating `global, sliding, sliding, mamba` layer motif
- local sliding-window attention layers
- SwiGLU feed-forward layers
- tied token embeddings and LM head (R2 rebalance: the saved vocab budget
  funds extra depth)
- the frozen `voidful/PangolinTokenizer` vocabulary (114944 padded entries)
- incremental decoding with a hybrid KV/conv-state cache (rolling window for
  sliding layers, O(1) Mamba steps)
- optional multi-token prediction loss for training
- optional QK logit clipping and learnable attention sink (off in the shipped
  R2 configs, matching the validated upstream recipe)
- an optional `mamba_ssm` GPU path for Megatron-compatible Mamba2 scan kernels,
  with a self-contained PyTorch fallback when those kernels are unavailable

The 300M config (20 layers, 8K context) is the proxy model family used for
systems validation. The 1B config (28 layers, 256K context) is the target
family configuration. The 1B 1M config keeps the same weights and enables
linear RoPE scaling x4 from the 256K base for inference-time extrapolation
experiments.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_300m()
model = BarbetForCausalLM(config)
```

## Hugging Face Loading

After converted `safetensors` and the remote-code files are uploaded to a
Hugging Face model repository, the model can be loaded with:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("voidful/barbet-1b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("voidful/barbet-1b-base", trust_remote_code=True)
```

The config files under `configs/` already include the `auto_map` fields required
for remote-code loading.

## Checkpoint Conversion

Production Megatron `torch_dist` checkpoints can be converted with:

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/megatron/checkpoint_dir \
  --output-dir /path/to/hf_export \
  --force
```

The converter exports the main causal-LM path to `model.safetensors`. Megatron
MTP auxiliary heads are training-only and are intentionally not exported.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Transformers Usage](docs/transformers_usage.md)
- [Checkpoint Conversion](docs/checkpoint_conversion.md)
- [Long Context](docs/long_context.md)
- [Development](docs/development.md)

## Current Limitations

- CPU-only Mamba uses the PyTorch fallback. For closest Megatron decode parity,
  install `mamba_ssm` and run on CUDA so the model uses the fused Mamba2 scan
  and gated RMSNorm path.
- The bundled PyTorch reference path can express the 1M RoPE extension, but
  practical 1M prefill still needs an optimized external long-context runtime.
  Global attention layers are quadratic without such a runtime.
