# Configuration

Barbet uses `BarbetConfig`, a standard Transformers `PretrainedConfig`.

## Presets

| Config | Layers | Hidden | FFN | Heads | KV Heads | Context | Window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `barbet_300m` | 20 | 1024 | 2816 | 8 | 2 | 8192 | 2048 |
| `barbet_1b` | 28 | 1536 | 5120 | 16 | 2 | 262144 | 8192 |
| `barbet_1b_1m` | 28 | 1536 | 5120 | 16 | 2 | 1048576 | 8192 |

Both shipped configs currently use:

- `vocab_size=114944`
- `head_dim=128`
- `rope_theta=10000000`
- `rms_norm_eps=1e-6`
- tied embeddings and LM head (`tie_word_embeddings=true`)
- `qk_logit_clip=false` and `attention_sink=false` (matching the validated
  upstream R2 recipe)

The vocabulary size is the Megatron-padded size for the frozen
`voidful/PangolinTokenizer`: the base BPE vocab is 114688, special tokens run
to id 114821 (effective size 114822), and the embedding is padded to 114944 (a
multiple of 128). The canonical token ids are baked into the configs:

- `unk_token_id=114688` (`<unk>`)
- `bos_token_id=114689` (`<s>`)
- `eos_token_id=114690` (`</s>`)
- `pad_token_id=114691` (`<pad>`)

If the tokenizer vocabulary ever changes, update both config files and
regenerate any converted checkpoints.

## Files

- `configs/barbet_300m/config.json`
- `configs/barbet_1b/config.json`
- `configs/barbet_1b_1m/config.json`

Each config contains:

```json
"auto_map": {
  "AutoConfig": "configuration_barbet.BarbetConfig",
  "AutoModel": "modeling_barbet.BarbetModel",
  "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM"
}
```

This is required for Hugging Face remote-code loading.

## Regenerating Configs

```bash
PYTHONPATH=src python scripts/write_model_configs.py
```

This rewrites the config folders from the Python factory methods:

- `BarbetConfig.barbet_300m()`
- `BarbetConfig.barbet_1b()`
- `BarbetConfig.barbet_1b_1m_extension()`

## Important Fields

`global_attention_layers`

Layers that use full causal attention.

`mamba_layers`

Layers that use the Mamba-style mixer.

`sliding_window_size`

Window size for all non-global, non-Mamba layers.

`rope_scaling`

Optional RoPE scaling metadata for research extensions, structured as
`{"type", "factor", "original_context_length"}`. The default 300M and 1B
configs do not use scaling. `BarbetConfig.barbet_1b_1m_extension()` mirrors the
upstream 1M research config with linear scaling factor 4.0 from the 256K base.
Only linear scaling affects the bundled PyTorch forward path; `yarn` and
`longrope` entries are metadata for external runtimes.

The 1M preset is an inference-time extrapolation config for the 256K-native 1B
weights. It should not be described as native 1M pretraining.

`qk_logit_clip`, `qk_clip_alpha`, `qk_clip_threshold`, `attention_sink`

Optional attention stabilizers from the long-term design. Both shipped R2
configs keep them disabled because the corresponding upstream Transformer
Engine features are compatibility-gated.

`mtp_enabled`, `mtp_offsets`, `mtp_loss_weights`

Controls auxiliary multi-token prediction training loss. Inference does not
require MTP outputs.
