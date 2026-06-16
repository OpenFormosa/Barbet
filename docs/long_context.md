# Long Context

Barbet 1B targets a 256K product context before any 1M extension. Barbet 300M
is a short-context proxy for systems validation.

## 256K Base (1B)

The 1B config sets:

```json
"max_position_embeddings": 262144
```

The current training curriculum is:

```text
32K -> 64K -> 128K -> 256K
```

Sliding attention layers keep a local window of 8192 tokens. Global attention
layers remain full causal attention.

## 8K Proxy (300M)

The 300M R2 config trains at 8K context with a 2048-token sliding window:

```json
"max_position_embeddings": 8192,
"sliding_window_size": 2048
```

## 1M Extension

The intended 1M path is not normal pretraining. It is a guarded research
extension from a validated 256K base.

The planned direction is:

```text
256K base -> linear RoPE x4 metadata -> external sparse/compressed-memory runtime
```

A native full-attention 1M run is not a default config because global attention
layers would still be expensive and could create misleading systems results.

## RoPE Scaling

For a 1M research extension from 256K, linear RoPE scaling uses:

```json
"rope_scaling": {
  "type": "linear",
  "factor": 4.0,
  "original_context_length": 262144
}
```

This metadata is supported by `BarbetConfig`, but the shipped 300M and 1B base
configs do not enable scaling. `BarbetConfig.barbet_1b_1m_extension()` produces
the 1M research configuration (`max_position_embeddings=1048576` with the
linear scaling block above), mirroring
`configs/model/open_formosa_1b_r2_1m_extension.yaml` upstream. The bundled
PyTorch forward applies linear position scaling; the upstream CSA/HCA-lite
compressed-memory mode remains the intended route for practical 1M
experiments.

The generated repository config is:

```text
configs/barbet_1b_1m/config.json
```

It is weight-compatible with the 256K Barbet 1B checkpoint because RoPE does
not add learned position parameters. The tradeoff is runtime: without a sparse,
DCA/YaRN-style, or compressed-memory engine, the global attention layers still
make 1M prefill impractical for normal Transformers inference.
