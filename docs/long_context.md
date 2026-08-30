# 長上下文

Hugging Face 上目前發布的 `OpenFormosa/barbet-1b-base` 是 `final-global-barbet-iter7008-stable-1m-v11`，原生 context length 為 1,048,576 tokens。它實際經過 exact-1M continued pretraining，正式 release config 不使用線性 RoPE scaling。

Repository 內的 `barbet_1b` 與 `barbet_1b_1m` 則是原始 R2 的 legacy presets，保留供舊版研究重現。不要把 legacy 1M extrapolation preset 誤認為目前 Hugging Face 權重隨附的 native-1M config。

## 目前正式 release

| 項目 | 值 |
| --- | ---: |
| Checkpoint | `final-global-barbet-iter7008-stable-1m-v11` |
| Native context | 1,048,576 |
| `rope_theta` | 10,000,000 |
| `rope_scaling` | `null` |
| Fresh exact-1M rows | 140/140 finite |
| Effective long-task families | 6/7 |
| Base-model BPB retention | 6/6 buckets |

完整架構演進、訓練 token ledger、能力評估與限制，請見[〈Barbet 1B Base：我們如何把 256K 外推研究模型變成原生 stable 1M〉](barbet_1b_native_1m.md)。

## Legacy presets

### 各設定的上下文長度

| 設定 | 最大上下文長度 | 滑動視窗 |
| --- | ---: | ---: |
| `barbet_1b` | 262144 | 8192 |
| `barbet_1b_1m` | 1048576 | 8192 |

### Legacy 1M 外推設定

Repository 內的 `configs/barbet_1b_1m/config.json` 不是 final iter7008 release config，而是從 256K 原始 R2 權重做推論外推的歷史設定。它透過線性 RoPE 縮放，把可用的上下文長度延伸到 1M：

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

因為 RoPE 不會額外增加學到的位置參數，所以這份 legacy 1M 設定與 256K 的原始 Barbet R2 權重相容，可以直接共用同一份權重。正式 iter7008 則應直接載入 Hugging Face repository 隨權重發布的 `config.json`。

## 使用須知

無論使用 legacy extrapolation preset 或正式 native-1M checkpoint，實務上執行 1M 都需要最佳化的長上下文環境、足夠 GPU memory 與 context parallelism。全域注意力在這個長度下成本仍然很高，用一般單卡 Transformers 推論不容易實際跑到 1M。
