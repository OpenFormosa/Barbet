# 開發

## 安裝

```bash
pip install -e ".[dev]"
```

## 測試

```bash
pytest -q
```

## 簡易驗證

可以用一個極小的設定快速確認模型能正常建立與前向傳遞：

```python
import torch
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=128,
    sliding_window_size=16,
    global_attention_layers=[0],
    mamba_layers=[2],
    pad_token_id=3,
    bos_token_id=1,
    eos_token_id=2,
    unk_token_id=0,
)
model = BarbetForCausalLM(config)
ids = torch.randint(0, 128, (1, 8))
out = model(input_ids=ids, labels=ids)
assert out.logits.shape == (1, 8, 128)
```
