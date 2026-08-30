# Barbet

Barbet 是 Barbet 因果語言模型系列的 Hugging Face Transformers 實作。Hugging Face 上目前發布的 `OpenFormosa/barbet-1b-base` 是 `final-global-barbet-iter7008-stable-1m-v11`：一個約 1.119B parameters、原生支援 1,048,576-token context 的 causal base model。這個 repository 也保留原始 R2 的 256K 與 1M RoPE 外推設定，供舊版研究重現。
<p align="center">
  <a href="docs/en/README_en.md">English README
</p>
<p align="center">
  <a href="https://huggingface.co/OpenFormosa/barbet-1b-base">
    <img src="https://img.shields.io/badge/Hugging%20Face-barbet--1b--base-FFD21E?logo=huggingface&logoColor" alt="Hugging Face Barbet 1B Base">
  </a>
  <img src="https://img.shields.io/badge/Native%20Context-1M-orange?logo=openai&logoColor=white" alt="Native Context 1M">
  
</p>
<p align="center">
<img src="https://img.shields.io/badge/Python-%3E%3D3.10-blue?logo=python&logoColor" alt="Python >=3.10">
<a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-yellow?logo=apache&logoColor" alt="Apache 2.0">
  </a>
<p align="center">

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

Barbet 是一個 decoder-only 的因果語言模型，因此有以下特點：

- 使用固定的 `OpenFormosa/PangolinTokenizer` 詞彙表
- 詞嵌入與 LM head 共用權重
- 支援逐步解碼（incremental decoding）的混合式快取，生成長序列時更省記憶體

目前 Hugging Face release 與 repository 內建 presets 的關係如下：

| 設定 | 用途 | 上下文長度 |
| --- | --- | ---: |
| `OpenFormosa/barbet-1b-base` | `iter7008` 正式權重與隨附 release config | **原生 1M** |
| `configs/barbet_1b/` | 原始 R2／legacy preset | 原生 256K |
| `configs/barbet_1b_1m/` | 原始 R2 的推論外推研究 preset | 1M RoPE 外推 |

正式 `iter7008` 實際在 exact 1M sequences 上 continued-pretrain；它不是把 legacy 256K 權重只靠 RoPE scaling 拉到 1M。完整訓練路徑、能力 gate 與限制見 OpenFormosa 官網的[〈Barbet 1B Base 的原生 1M：從位置外推到可驗證的遠距資訊使用〉](https://openformosa.com/blog/2026/08/30/barbet-1b-native-1m/)。

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

當轉換後的 `safetensors` 與 remote code 檔案上傳到 Hugging Face 模型庫後，透過下列載入：

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("OpenFormosa/barbet-1b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OpenFormosa/barbet-1b-base", trust_remote_code=True)
```

`configs/` 底下的設定檔已經包含 remote code 載入所需的 `auto_map` 欄位。

## 檢查點轉換

在終端機可以透過以下指令把 Megatron `torch_dist` 檢查點轉換成 Hugging Face 格式：

```bash
python scripts/convert_torch_dist_to_hf.py \
  --checkpoint /path/to/megatron/checkpoint_dir \
  --output-dir /path/to/hf_export \
  --force
```

轉換器會把主要的 causal-LM 權重輸出成 `model.safetensors`

## 文件

- [設定](docs/configuration.md)
- [Transformers 使用方式](docs/transformers_usage.md)
- [檢查點轉換](docs/checkpoint_conversion.md)
- [長上下文](docs/long_context.md)
- [Barbet 1B Base 的原生 1M：完整訓練與評測報告](https://openformosa.com/blog/2026/08/30/barbet-1b-native-1m/)
- [開發](docs/development.md)
- [授權](LICENSE)
- [Model card](model_cards/barbet-1b-base/README.md)

## 使用限制

- 只有 CPU 時，Mamba 會使用 PyTorch 後備路徑。若要得到最接近原始模型的解碼結果，請安裝 `mamba_ssm` 並在 CUDA 上執行。
- 正式模型雖已在 exact 1M 上完成訓練與評估，實務上執行 1M 仍需要足夠 GPU memory、context parallelism 與相容的最佳化 kernels；單張消費級 GPU 通常不可行。
