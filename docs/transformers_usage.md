# Transformers 使用方式

Barbet 透過 Hugging Face Hub 的自訂程式（remote code）載入。使用者用標準的 `AutoConfig`、`AutoTokenizer`、`AutoModelForCausalLM`、`generate()` 與 `pipeline()` 就能使用，只要多加一個 `trust_remote_code=True`。

## 支援範圍

以下項目已用 `OpenFormosa/barbet-1b-base` 的真實權重實測，環境是只有 CPU、沒有安裝 `mamba_ssm`：

| 項目 | transformers 4.57.6 | transformers 5.24.1 |
| --- | :---: | :---: |
| `AutoConfig` / `AutoModelForCausalLM`（`trust_remote_code=True`） | ✓ | ✓ |
| `AutoTokenizer`（直接從模型 repo 載入） | ✓ | ✓ |
| safetensors 載入，BF16 權重與 FP32 的 Mamba 狀態參數都保持原精度 | ✓ | ✓ |
| `generate()`，開啟與關閉快取的結果逐 token 相同 | ✓ | ✓ |
| 左側補齊（left padding）的批次生成，與逐筆生成結果相同 | ✓ | ✓ |
| `pipeline("text-generation")` | ✓ | ✓ |
| `save_pretrained()` 後重新載入，權重完全相同 | ✓ | ✓ |

還沒有實測的部分：CUDA 上的 `mamba_ssm` kernel 路徑、FlexAttention 長序列路徑、1M 長度的實際推論、多 GPU `device_map`。這些路徑的程式沿用 Hub 上已發布的版本，本 repository 的測試只在 CPU 上跑。

建議版本：`transformers>=4.57`、`torch>=2.3`。更舊的 transformers 可能不會保留 Mamba 狀態參數的 FP32 精度。

## 從 Hub 載入

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "OpenFormosa/barbet-1b-base"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype="auto",  # BF16 權重，Mamba 的 A_log 與 D 自動保持 FP32
)
```

分詞器檔案和權重放在同一個模型 repo，所以 `AutoTokenizer` 直接指向模型 repo 即可。分詞器預設不加 `<s>`，一般續寫照預設使用即可。

`trust_remote_code=True` 會執行模型 repo 裡的 `configuration_barbet.py`、`modeling_barbet.py` 與 `megatron_long_context.py`。正式部署時，請用 `revision=` 固定到一個你審閱過的 commit。

## 文字生成

```python
inputs = tokenizer("台灣的健保制度", return_tensors="pt")
ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

`generate()` 預設使用 Barbet 自己的混合式快取（`BarbetCache`）。注意力層存 K/V，Mamba 層存卷積尾端與 SSM 狀態，所以每生成一個 token 只算一個位置。傳入 `use_cache=False` 會每步重算整段，結果完全相同，只是比較慢。

設定檔已內建 PangolinTokenizer 的標準 token id（`eos_token_id=114690`、`pad_token_id=114691`），生成時的停止條件與補齊會自動和分詞器一致。

### 批次生成

批次生成請用左側補齊：

```python
tokenizer.padding_side = "left"
batch = tokenizer(["台灣的首都是", "今天天氣很好，我們一起去"], return_tensors="pt", padding=True)
out = model.generate(**batch, max_new_tokens=32, do_sample=False)
```

Mamba 層會把補齊位置的輸入與卷積輸出歸零，補齊的 token 不會寫進狀態。所以每一筆的結果和單獨生成時相同。

### Pipeline

```python
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(generator("台灣的首都是", max_new_tokens=20, do_sample=False)[0]["generated_text"])
```

## 硬體與加速

| 環境 | Mamba 路徑 | 注意力路徑 |
| --- | --- | --- |
| CPU，或沒有 `mamba_ssm` | PyTorch 逐步掃描（正確但慢） | PyTorch SDPA |
| CUDA + `mamba_ssm` | `mamba_chunk_scan_combined` kernel | SDPA；長序列走 FlexAttention |

沒有 `mamba_ssm` 時，程式會自動改用純 PyTorch 實作，不需要改任何設定。純 PyTorch 的 Mamba 掃描是逐 token 迴圈，適合短輸入與測試。長輸入請在 CUDA 上安裝相容版本的 `mamba_ssm`。

## 本機資料夾

把以下檔案放在同一個資料夾，就能用同樣的方式載入：

```text
config.json
generation_config.json
configuration_barbet.py
modeling_barbet.py
megatron_long_context.py
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

`model.save_pretrained()` 會自動把三個 `.py` 檔複製到輸出資料夾，不需要手動複製。

## 本機 Python 套件

不經過 Hub、直接從原始碼使用：

```bash
pip install -e ".[dev]"
```

```python
import torch
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_1b_base()  # 與 Hub 權重相同的架構
model = BarbetForCausalLM(config)
```

`configs/barbet_1b_base/config.json` 是與 Hub 權重一起發布的設定。`configs/barbet_1b/` 與 `configs/barbet_1b_1m/` 是早期 28 層 R2 架構的研究設定，和目前發布的權重不相容。

## 和 transformers 原生支援的關係

「原生支援」是指不需要 `trust_remote_code`，架構直接寫在 transformers 套件裡。Barbet 目前不是原生支援，原因如下：

1. transformers 現有的 Mamba2 混合式架構（Bamba、NemotronH、GraniteMoeHybrid、FalconH1、Zamba2）都沒有 QK RMSNorm。Barbet 的注意力層有 QK RMSNorm，所以不能把權重轉成現有架構而不改變輸出。
2. 要原生支援，就要向 transformers 上游新增一個 `barbet` 架構。這需要 modular 格式的實作、改用 transformers 的標準快取類別、完整測試與文件，並由 Hugging Face 維護者審查。審查時程不由我們決定。
3. 上游的整合測試需要公開的權重。`OpenFormosa/barbet-1b-base` 公開之前，上游合併不實際。

因此目前的做法是 Hub 自訂程式。對使用者來說，差別只有多一個 `trust_remote_code=True`。
