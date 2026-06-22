# 長上下文

Barbet 1B 的目標上下文長度是 256K，另外提供一個推論時的 1M 外推設定。

## 各設定的上下文長度

| 設定 | 最大上下文長度 | 滑動視窗 |
| --- | ---: | ---: |
| `barbet_1b` | 262144 | 8192 |
| `barbet_1b_1m` | 1048576 | 8192 |

## 1M 外推設定

1M 設定不是原生的長上下文預訓練，而是從 256K 的 1B 權重做推論時外推。它透過線性 RoPE 縮放，把可用的上下文長度延伸到 1M：

```json
"rope_scaling": {
  "type": "linear",
  "factor": 4.0,
  "original_context_length": 262144
}
```

可以用工廠方法取得這份設定：

```python
from barbet import BarbetConfig

config = BarbetConfig.barbet_1b_1m_extension()
print(config.max_position_embeddings)  # 1048576
```

對應的設定檔是：

```text
configs/barbet_1b_1m/config.json
```

因為 RoPE 不會額外增加學到的位置參數，所以 1M 設定與 256K 的 Barbet 1B 權重相容，可以直接共用同一份權重。

## 使用須知

實務上要跑到 1M 等級的長上下文，仍需要額外經過最佳化的長上下文執行環境。內建的 PyTorch 參考路徑雖然可以表達 RoPE 縮放，但全域注意力（global attention）層在這個長度下成本仍然很高，用一般的 Transformers 推論不容易實際跑到 1M。
