# Transformers Usage

## Local Python Package

```bash
pip install -e ".[dev]"
```

```python
import torch
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_300m()
model = BarbetForCausalLM(config)

input_ids = torch.randint(0, config.vocab_size, (1, 16))
outputs = model(input_ids=input_ids)
print(outputs.logits.shape)
```

## Loading a Config Folder

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "configs/barbet_300m",
    trust_remote_code=True,
)
```

When loading directly from a local config folder, make sure the folder also
contains the remote-code files if you want to instantiate `AutoModel`.

## Hugging Face Hub Layout

A model repository should contain at least:

```text
config.json
configuration_barbet.py
modeling_barbet.py
model.safetensors
```

For config-only publication, omit `model.safetensors`.

## Loading From Hub

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("voidful/PangolinTokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "voidful/barbet-1b-base",
    trust_remote_code=True,
)
```

For config-only inspection, or before HF weights are published:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("voidful/barbet-1b-base", trust_remote_code=True)
print(config.max_position_embeddings)
```

The Hub repo may expose the 1M extension as the default `config.json` and keep
the native 256K base config as `config_256k.json`. Both configs use the same
1B R2 weights; the 1M config only changes RoPE scaling metadata and maximum
context length.

## Generation

```python
prompt = "台灣的健保制度"
inputs = tok(prompt, return_tensors="pt")
ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(ids[0], skip_special_tokens=True))
```

`generate()` uses incremental decoding by default (`use_cache=True`): attention
layers cache K/V states (sliding layers keep only the local window) and
Mamba-style layers carry their causal-conv state, so each new token costs a
single-token forward instead of recomputing the full sequence. Pass
`use_cache=False` to force full recomputation; both paths produce identical
tokens.

The shipped configs carry the canonical PangolinTokenizer ids
(`eos_token_id=114690`, `pad_token_id=114691`), so generation stopping and
padding agree with the tokenizer without extra arguments.

## Saving

```python
model.save_pretrained("barbet-300m-local")
config.save_pretrained("barbet-300m-local")
```

Then copy `configuration_barbet.py` and `modeling_barbet.py` into the same
folder before using remote-code `AutoModel` loading.
