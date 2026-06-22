# Barbet

Barbet 是 Barbet 因果語言模型系列的 Hugging Face Transformers 實作。這個專案提供可透過 remote code 載入的模型類別，以及兩組設定預設：Barbet 1B，還有作為研究延伸用途的 Barbet 1B 1M。

本專案刻意保持輕量，只包含模型程式碼、設定檔，以及檢查點轉換工具。

## 內容

- `BarbetConfig`
- `BarbetModel`
- `BarbetForCausalLM`
- `configs/barbet_1b/config.json`
- `configs/barbet_1b_1m/config.json`
- 供 Hugging Face Hub 以 remote code 載入的檔案：
  - `configuration_barbet.py`
  - `modeling_barbet.py`

## 模型簡介

Barbet 是一個 decoder-only 的因果語言模型。對使用者來說，需要知道的重點是：

- 使用固定的 `openformosa/PangolinTokenizer` 詞彙表
- 詞嵌入與 LM head 共用權重
- 支援逐步解碼（incremental decoding）的混合式快取，生成長序列時更省記憶體

提供的兩組設定：

| 設定 | 用途 | 上下文長度 |
| --- | --- | ---: |
| Barbet 1B | 主要目標模型 | 256K |
| Barbet 1B 1M | 推論時的長上下文外推研究設定（與 1B 共用權重） | 1M |

## 快速開始

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from barbet import BarbetConfig, BarbetForCausalLM

config = BarbetConfig.barbet_1b()
model = BarbetForCausalLM(config)
```

## 從 Hugging Face 載入

當轉換後的 `safetensors` 與 remote code 檔案上傳到 Hugging Face 模型庫後，可以這樣載入：

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openformosa/barbet-1b-base", trust_remote_code=True)
```

`configs/` 底下的設定檔已經包含 remote code 載入所需的 `auto_map` 欄位。

## 檢查點轉換

可以用以下指令把 Megatron `torch_dist` 檢查點轉換成 Hugging Face 格式：

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/megatron/checkpoint_dir \
  --output-dir /path/to/hf_export \
  --force
```

轉換器會把主要的 causal-LM 權重輸出成 `model.safetensors`。

## 文件

- [設定](docs/configuration.md)
- [Transformers 使用方式](docs/transformers_usage.md)
- [檢查點轉換](docs/checkpoint_conversion.md)
- [長上下文](docs/long_context.md)
- [開發](docs/development.md)

## 使用限制

- 只有 CPU 時，Mamba 會使用 PyTorch 後備路徑。若要得到最接近原始模型的解碼結果，請安裝 `mamba_ssm` 並在 CUDA 上執行。
- 內建的 PyTorch 參考路徑雖然可以表達 1M 的 RoPE 延伸，但實務上要跑到 1M 等級的長上下文，仍需要額外經過最佳化的長上下文執行環境。
