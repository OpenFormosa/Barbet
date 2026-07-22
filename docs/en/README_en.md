# Barbet

Barbet is the Hugging Face Transformers implementation of the Barbet family of causal language models. This project provides model classes that can be loaded through remote code, along with two configuration presets: Barbet 1B and Barbet 1B 1M, which is intended for extended research use.

<p align="center">
  <a href="https://huggingface.co/OpenFormosa/barbet-1b-base">
    <img src="https://img.shields.io/badge/Hugging%20Face-barbet--1b--base-FFD21E?logo=huggingface&logoColor" alt="Hugging Face Barbet 1B Base">
  </a>
  <img src="https://img.shields.io/badge/Context%20Window-256k-orange?logo=openai&logoColor=white" alt="Context Window 256k">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%3E%3D3.10-blue?logo=python&logoColor" alt="Python >=3.10">
  <a href=".github/LICENSE.md">
    <img src="https://img.shields.io/badge/license-Apache%202.0-yellow?logo=apache&logoColor" alt="Apache 2.0">
  </a>
</p>

This project is intentionally lightweight and contains only the model implementation, configuration files, and checkpoint conversion tools.

## Contents

* `BarbetConfig`
* `BarbetModel`
* `BarbetForCausalLM`
* `configs/barbet_1b/config.json`
* `configs/barbet_1b_1m/config.json`
* Files for loading the model from the Hugging Face Hub through remote code:

  * `configuration_barbet.py`
  * `modeling_barbet.py`

## Model Overview

Barbet is a decoder-only causal language model. The key characteristics are:

* Uses the fixed `openformosa/PangolinTokenizer` vocabulary
* Shares weights between the token embeddings and the LM head
* Supports a hybrid cache for incremental decoding, reducing memory usage when generating long sequences

The model provides two configuration presets:

| Configuration |Purpose | Context Length |
| ------ | --------- | --------: |
| Barbet 1B     | Primary target model    |  256K |
| Barbet 1B 1M  | Research configuration for long-context extrapolation during inference, sharing the same weights as Barbet 1B |   1M |

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_1b()
model = BarbetForCausalLM(config)
```

## Loading from Hugging Face

After the converted `safetensors` file and remote code files have been uploaded to a Hugging Face model repository, load the model as follows:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
```

The configuration files under `configs/` already include the `auto_map` fields required for remote code loading.

## Checkpoint Conversion

Use the following command to convert a Megatron `torch_dist` checkpoint into Hugging Face format:

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/megatron/checkpoint_dir \
  --output-dir /path/to/hf_export \
  --force
```

The converter exports the main causal language model weights as `model.safetensors`.

## Documentation

* [Configuration](docs/configuration.md)
* [Transformers Usage](docs/transformers_usage.md)
* [Checkpoint Conversion](docs/checkpoint_conversion.md)
* [Long Context](docs/long_context.md)
* [Development](docs/development.md)
* [License](.github/LICENSE.md)
* [Model Card](model_cards\barbet-1b-base\README.md)

## Limitations

* When running on CPU only, Mamba uses the PyTorch fallback implementation. To obtain decoding results that most closely match the original model, install `mamba_ssm` and run the model on CUDA.
* Although the built-in PyTorch reference implementation can represent the 1M-token RoPE extension, running at the 1M-context scale in practice still requires an additionally optimized long-context execution environment.
