# Checkpoint Conversion

This repository provides the Hugging Face model class and a production
Megatron `torch_dist` to HF `safetensors` converter for Barbet 1B R2.

## Why Conversion Is Needed

The training stack uses Megatron HybridModel with a mixed attention/Mamba layer
schedule. Its checkpoint layout differs from the Hugging Face module layout in
this repository. The upstream reference model also fuses the key/value
projection into one `kv` matrix, while this implementation keeps separate
`k_proj`/`v_proj` (standard HF convention).

Conversion must map:

- token embeddings (the LM head is tied to them in R2; converted checkpoints
  store only `model.embed_tokens.weight` and `lm_head.weight` is re-tied at
  load time);
- final RMSNorm;
- attention query/key/value/output projections;
- QK RMSNorm parameters;
- SwiGLU gate/up/down projections;
- Mamba-style mixer parameters;
- training-only MTP auxiliary heads are intentionally omitted from inference
  exports.

## Command

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/open_formosa_1b_checkpoint \
  --output-dir /path/to/barbet-1b-hf \
  --force
```

The `--checkpoint` path can be either the checkpoint parent directory that
contains `latest_checkpointed_iteration.txt` or a concrete `iter_0000xxx`
directory.

## Expected Output

A converted Hugging Face checkpoint should contain:

```text
config.json
config_1m_extension.json
configuration_barbet.py
modeling_barbet.py
model.safetensors
conversion_report.json
```

## Conversion Mapping

The converter maps Megatron HybridModel residual modules back into 28 logical
HF decoder blocks:

- Megatron `decoder.layers.2*i` -> HF token mixer for logical layer `i`.
- Megatron `decoder.layers.2*i+1` -> HF MLP for logical layer `i`.
- Grouped Megatron `linear_qkv.weight` is split from interleaved GQA groups
  into HF `q_proj`, `k_proj`, and `v_proj`.
- Megatron SwiGLU `linear_fc1.weight` is split into HF `gate_proj` and
  `up_proj`.
- Megatron Mamba2 `z/x/B/C/dt`, conv, `A_log`, `D`, `dt_bias`, gated norm,
  and output projection tensors map directly onto the HF Mamba mixer.

It should load with:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/converted-barbet",
    trust_remote_code=True,
)
```

## Conversion Gates

A conversion run should verify:

- all expected HF keys are present;
- no unexpected Megatron shards are silently ignored;
- embedding and LM head shapes match `vocab_size`;
- 300M and 1B configs produce the expected parameter shapes;
- logits from a tiny deterministic fixture match before and after conversion
  where a reference path is available;
- `save_pretrained` and `from_pretrained` both work;
- `generate()` smoke test runs.

## Current Status

The HF implementation has been smoke-tested for:

- construction;
- forward pass;
- training loss;
- `save_pretrained`;
- remote-code `AutoConfig`;
- remote-code `AutoModelForCausalLM`;
- `generate()`.

The production converter is implemented in
`scripts/convert_torch_dist_to_hf.py`. The validated 1B export uses native 256K
as `config.json` and writes the 1M extrapolation metadata to
`config_1m_extension.json` so default loading preserves native-checkpoint decode
behavior.
