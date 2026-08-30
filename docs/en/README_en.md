# Barbet

> **Release update (2026-08-30):** the current `OpenFormosa/barbet-1b-base` checkpoint is `final-global-barbet-iter7008-stable-1m-v11`, trained and evaluated at a native 1,048,576-token context. The 256K / 1M-RoPE-scaling presets documented below are retained as legacy R2 research configs. See the detailed [native-1M training article](../barbet_1b_native_1m.md).

Barbet is the Hugging Face Transformers implementation of the Barbet family of causal language models. The current Hugging Face release is native 1M; this repository also retains two legacy R2 configuration presets for reproducibility.

<p align="center">
  <a href="https://huggingface.co/OpenFormosa/barbet-1b-base">
    <img src="https://img.shields.io/badge/Hugging%20Face-barbet--1b--base-FFD21E?logo=huggingface&logoColor" alt="Hugging Face Barbet 1B Base">
  </a>
  <img src="https://img.shields.io/badge/Native%20Context-1M-orange?logo=openai&logoColor=white" alt="Native Context 1M">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%3E%3D3.10-blue?logo=python&logoColor" alt="Python >=3.10">
  <a href="../../LICENSE">
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

* Uses the fixed `OpenFormosa/PangolinTokenizer` vocabulary
* Shares weights between the token embeddings and the LM head
* Supports a hybrid cache for incremental decoding, reducing memory usage when generating long sequences

The current release and the repository presets have different roles:

| Configuration |Purpose | Context Length |
| ------ | --------- | --------: |
| `OpenFormosa/barbet-1b-base` | Current iter7008 release | Native 1M |
| `configs/barbet_1b/` | Legacy R2 preset | Native 256K |
| `configs/barbet_1b_1m/` | Legacy R2 inference extrapolation preset | 1M RoPE extrapolation |

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

config = AutoConfig.from_pretrained("OpenFormosa/barbet-1b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OpenFormosa/barbet-1b-base", trust_remote_code=True)
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

* [Configuration](../configuration.md)
* [Transformers Usage](../transformers_usage.md)
* [Checkpoint Conversion](../checkpoint_conversion.md)
* [Long Context](../long_context.md)
* [Native-1M Training Article](../barbet_1b_native_1m.md)
* [Development](../development.md)
* [License](../../LICENSE)
* [Model Card](../../model_cards/barbet-1b-base/README.md)

## Limitations

* When running on CPU only, Mamba uses the PyTorch fallback implementation. To obtain decoding results that most closely match the original model, install `mamba_ssm` and run the model on CUDA.
* The stable-1M designation is a scoped likelihood-based capability claim, not universal comprehension of arbitrary million-token documents; aggregation remains unproven.
* Running a true 1M context still requires sufficient GPU memory, context parallelism, and compatible optimized kernels; a single consumer GPU is generally insufficient.
