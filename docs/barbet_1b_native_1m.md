# Barbet 1B Base：我們如何把 256K 外推研究模型變成原生 stable 1M

2026 年 8 月 30 日 · pretrain、model、long-context、barbet

> 摘要：2026 年 6 月發表的 [Barbet 1B Base 介紹](https://openformosa.com/blog/2026/06/21/barbet-1b-base/) 嚴格區分了「256K 原生訓練」與「1M RoPE 外推」。兩個月後，我們完成了同一 Barbet lineage 的原生 1M continued pretraining：最終 checkpoint `final-global-barbet-iter7008-stable-1m-v11` 不再使用線性 RoPE 外推，實際在 exact 1,048,576-token sequences 上接受訓練，並以 fresh-loaded、適合 base model 的 frozen likelihood gate 證明它能在六類長上下文任務中利用遠距資訊，同時保留六類原始語言模型能力。本文說明這條路徑，也明確揭露仍未突破的 aggregation 限制。

## 本文重點

- **1M 不再只是設定值。** 最終 checkpoint 的 `max_position_embeddings=1,048,576`、`rope_theta=10,000,000`、`rope_scaling=null`，而且 selected lineage 實際消耗 3,925,868,544 個 exact-1M physical training tokens。
- **沒有換掉 Barbet。** Tokenizer、hidden size、Mamba blocks 與 causal base-LM objective 保持在同一 lineage；長距 token-mixing 路徑則從 `[G,S,S,M]×7` 漸進擴充為 `[G,G,G,M]×7+G`。
- **不是只把文件拉長。** 訓練資料依序覆蓋 retrieval、variable binding、ordering、multi-key、短鏈組合與不同 evidence depths；當 1M 診斷顯示模型會 copy 卻不一定會 compute 時，我們回到 512–8K 隔離 operator acquisition 與 remote binding。
- **能力與 runtime 分開驗證。** Fresh exact-1M evaluation 的 140/140 rows 全部 finite；七類 long-context tasks 有 6/7 的 paired-bootstrap CI95 lower bound 大於零；六類 base-model BPB retention 全部通過。
- **Stable 1M 是有界結論。** Exact NIAH、opaque NIAH、multi-key、ordering、variable tracking 與 three-hop chain 成立；aggregation 尚未證明。這不是任意 1M 文件理解、free-generation、instruction following 或安全部署的保證。

## 從「1M 外推」到「原生 1M」

最初發布的 Barbet R2 是一個 28 層的混合式 causal language model，原生訓練長度到 262,144 tokens。當時提供的 1M 設定是：

```json
{
  "max_position_embeddings": 1048576,
  "rope_scaling": {
    "type": "linear",
    "factor": 4.0,
    "original_context_length": 262144
  }
}
```

這個設定能讓同一組權重在推論時表達 1M positions，但不能證明模型曾在 1M 上更新權重，也不能證明它會使用第 900K token 附近的證據。

最終 stable-1M release 改成：

```json
{
  "max_position_embeddings": 1048576,
  "rope_theta": 10000000.0,
  "rope_scaling": null
}
```

兩者的差別不是 JSON 少了一段而已。新的 checkpoint 實際經歷 exact-1M forward、backward、optimizer update、checkpoint reload 與 fresh capability evaluation。對我們而言，「原生 1M」必須同時回答三個問題：

1. **物理可行性：** 模型能否完成 exact 1,048,576-token computation，而不出現 OOM、NaN 或 non-finite score？
2. **能力：** 正確的遠距 evidence 是否真的提高正確 continuation 的 likelihood？
3. **保留：** 長上下文 continued pretraining 是否破壞原始的中文、多語、英文、數學與程式碼建模能力？

只通過第一項，最多只能說「1M 能跑」。Stable 1M 必須三項一起成立。

## 起點：先保留原始 Barbet 函數

主線從 Role-B native all-S checkpoint 開始：

```text
[G, S, S, M] × 7
```

其中 `G` 是 Global Attention，`S` 是 8K Sliding Attention，`M` 是 Mamba2。開始長上下文訓練前，original Hugging Face checkpoint 到 native Megatron Role-B 的轉換已在 4K／8K selected raw-FP32 logits 上證明 exact。這件事很重要：如果起點本身已經漂移，後面任何「長能力提升」都可能只是 conversion artifact。

早期研究也測試過 KDA conversion，但 pure-KDA 的原始能力 retention 明顯不足，因此 KDA 沒有進入 1M release 的 critical path。最終 Hugging Face checkpoint 不包含 KDA、QSA、external-memory controller 或其他 test-time memory module。

## 架構演進：讓遠距 evidence 有更直接的路徑

原始 `[G,S,S,M]×7` 每四層只有一個 Global Attention。這對 256K 是務實的成本折衷，但在 1M 上，兩個 Sliding slots 會使任意遠距 token 的交互路徑變長。為改善直接存取，我們沿同一 Barbet lineage 漸進擴充 Global Attention：

```text
[G,S,S,M]×7
    │  第二個 Sliding slot → Global
    ▼
[G,G,S,M]×7
    │  第三個 Sliding slot → Global
    ▼
[G,G,G,M]×7
    │  identity-only 新增 final Global layer
    ▼
[G,G,G,M]×7+G
```

最終模型規格如下：

| 項目 | 最終值 |
| --- | ---: |
| Stored parameters（不重複 tied LM head） | 1,118,799,096 |
| Hidden size | 1,536 |
| FFN size | 5,120 |
| Logical layers | 29 |
| Global Attention layers | 22 |
| Mamba2 layers | 7 |
| Attention heads / KV heads | 16 / 2 |
| Vocabulary size | 114,944 |
| Native context | 1,048,576 |

最後一層 Global Attention 先以 identity-only 形式物化，因此插入動作本身消耗 0 training tokens；之後才進入 acquisition 與 exact-1M training。這次物化曾重設 iteration counter，所以名稱中的 `iter7008` 不代表整條 lineage 只有 7,008 updates。從 Role-B 起算，release ancestry 實際包含 11,715 個 accepted optimizer updates。

這仍然是 Barbet，而不是把 checkpoint 換成另一個模型家族：PangolinTokenizer、hidden size、Mamba2 blocks、tied embedding／LM head 與 causal base-model objective 都延續下來。改變的是遠距 token mixing 的容量。

## 訓練路徑：逐步拉長，而不是一步跳到 1M

歷史 R2 repository 記錄了約 150B-token 的原始 recipe budget：8K general pretraining、繁中 midtraining，以及 32K、64K、128K、256K 的 progressive extension。不過歷史資料沒有把每個 phase 的 completion 與發布 Role-B checkpoint 綁成單一 completion receipt，因此這 150B 應寫成 documented recipe budget，而不是 checkpoint-bound exact consumed total。

Role-B 之後的 continued pretraining 則有完整 selected-lineage ledger：

| Physical sequence length | Audited physical tokens |
| ---: | ---: |
| 512 | 144,703,488 |
| 1K | 25,165,824 |
| 2K | 25,165,824 |
| 4K | 25,165,824 |
| 8K | 427,819,008 |
| 64K | 100,663,296 |
| 128K | 305,135,616 |
| 256K | 268,435,456 |
| 512K | 268,435,456 |
| exact 1M | **3,925,868,544** |
| **合計** | **5,516,558,336** |

完整 selected lineage 有 42 個 accepted optimizer stages；其中 22 個 stages、3,744 updates 直接使用 exact 1,048,576-token sequences。Zero-update pre-forward failures、evaluation、export、reload、identity materialization 與未被選入 release ancestry 的 branches 都不計入這個數字。

### 第一段：64K 與 128K 的能力校準

第一個 accepted 64K stage 使用 30% short replay、45% coherent Taiwan Books、25% structured base-LM data，執行 96 updates、100,663,296 tokens。Natural-book paired diagnostic 出現正向 distant-context signal，但 synthetic probe 沒有一致提升，因此當時只宣稱 task-specific evidence，沒有過早稱為 general stable 64K。

128K 階段進一步通過 training health、full-state reload 與六桶 retention；然而相較 64K parent 的 incremental distant-context effect 信賴區間穿越零。這個結果建立了一條後來非常重要的原則：

> 更低的 BPB、更長的可執行序列，並不自動等於更強的遠距資訊使用能力。

### 第二段：256K → 512K → exact 1M

Selected 128K parent 依序進入：

- 256K：1,024 updates、268,435,456 tokens；
- 512K：512 updates、268,435,456 tokens；
- 第一段 exact 1M：1,024 updates、1,073,741,824 tokens。

這一段建立了 physical exact-1M runtime。接著的 exact-1M correctives 不只是再塞長文件，而是分別處理：

- variable binding；
- arithmetic dependency 與 edge cases；
- ordered state；
- multi-field evidence；
- middle-position evidence；
- all-depth 與 far-distance coverage；
- aggregation 與 structured dependency。

之後再配合 `[G,G,S,M]×7`、`[G,G,G,M]×7` 與 final Global 的架構轉換 stages，讓新增的全域路徑在 1M 上取得實際功能。

## 為什麼已經在跑 1M，後來又回到 512–8K？

早期 exact-1M runs 已證明 forward／backward 可以完成，也在多個 retrieval tasks 上出現正向 signal。但最主要的缺口是 aggregation 與 state update：模型有時能找到並複製遠處的值，卻沒有可靠地把同一份 evidence 帶入計算。

如果繼續只在 1M 上追加訓練，失敗原因會混在一起：

```text
找不到 evidence
    或
找到了，但不會做 operator
    或
兩者單獨會，卻無法在遠距條件下組合
```

因此主線回到 512、1K、2K、4K、8K，逐層隔離：

1. local aggregation／state update 是否能取得；
2. literal remote copy 是否能完成；
3. 同一個 operator 能否綁定到 8K 遠距 evidence；
4. 完成後再回 exact 1M 做 closeout。

這些 frozen diagnostics 給出了一個很清楚的負面結果：**copying 不是 computation**。某些 stages 的 remote-copy controls 幾乎全對，但 remote aggregation／state update 仍失敗。這也是我們沒有用單一 NIAH 分數宣稱「模型已理解 1M」的原因。

## 最後一段 consolidation

Final release stage 從 immutable iter6240 執行到 iter7008：

| 設定 | 值 |
| --- | ---: |
| Sequence length | 8,192 |
| Optimizer updates | 768 |
| Physical tokens | 100,663,296 |
| Global batch | 16 |
| Tokens per update | 131,072 |
| Runtime | TP2 × CP4 × DP1 |
| Precision | BF16 |
| Optimizer | AdamW |
| Betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Weight decay / gradient clip | `0.1` / `1.0` |
| LR | stage-local cosine `1.2e-6 → 3e-7` |
| Warmup | 16 updates |
| Seed | 17 |

資料 mix 為：

- 25% replay：限制一般能力遺忘；
- 50% strict-digit remote compute binding：讓遠距 evidence 必須影響 continuation；
- 12.5% local sampler repair：維持基礎 operator；
- 12.5% literal remote 8K：作為 retrieval／copy control。

這一段是 ordinary all-token causal next-token cross-entropy，不使用 answer-only loss、teacher logits、distillation、SFT 或 RLHF。Training health 為 768/768 updates、0 skipped、0 NaN、0 OOM/fatal，terminal checkpoint 正常保存。

最後幾個 stages 使用 8K，不代表模型退回 8K。Exact-1M position exposure 與全域路徑已由前面約 3.926B exact-1M tokens 建立；短距 stages 的用途是用較低成本修補 operator binding。完成後，release gate 重新 fresh-load 最終 checkpoint 並在 exact 1M 上評估，以檢查是否遺忘。

## 我們如何判定「模型真的會用 1M」？

### 第一層：物理 exact 1M

Fresh-loaded iter7008 在 8 個 shards 上完成 140/140 個 exact 1,048,576-token rows；所有 target NLL 都是 finite，沒有 OOM 或 NaN。

這證明 runtime 與 checkpoint 是真的，但還不夠證明 capability。

### 第二層：coherent evidence 必須造成可歸因的 likelihood 改變

每個 evaluation row 都有一組 matched contexts：

- **coherent context**：保留正確的遠距 evidence；
- **ablated context**：長度與表面結構匹配，但移除或破壞關鍵 evidence。

模型是 plain causal base LM，評分使用 teacher-forced target-only NLL：

```text
evidence benefit = ablated target NLL − coherent target NLL
```

正值表示 coherent evidence 讓正確 continuation 更可能。這不是 chat prompting、multiple choice 或 greedy exact-generation。

每個 task family 有 20 rows，使用 2,000 次 paired bootstrap。Frozen release rule 要求七類 tasks 至少五類的 CI95 lower bound 大於零。結果如下：

| Exact-1M task | Mean nats / target token | 95% CI | 結果 |
| --- | ---: | ---: | ---: |
| Exact NIAH | +0.868817 | [+0.411734, +1.359182] | PASS |
| Opaque NIAH | +1.451789 | [+0.751489, +2.303071] | PASS |
| Multi-key | +1.041410 | [+0.494490, +1.671721] | PASS |
| Ordering | +1.160107 | [+0.710641, +1.667173] | PASS |
| Variable tracking | +0.249664 | [+0.075717, +0.456798] | PASS |
| Three-hop chain | +1.338904 | [+0.665179, +2.092832] | PASS |
| Aggregation | +0.028431 | [−0.019926, +0.080348] | **NOT PROVEN** |

最終為 6/7 effective。Aggregation 雖然平均值略為正，但信賴區間穿越零；evidence 位於 30% 與 50% depth 時，平均 effect 也仍為負，因此不能被 overall average 掩蓋。

### 第三層：原始 base-model retention

長上下文能力不能以犧牲原始語言模型為代價。Frozen retention suite 有 6,955 rows、3,000,079 model tokens，使用 bits per UTF-8 byte（BPB；越低越好）比較 iter7008 與 Role-B：

| Bucket | iter7008 BPB | Role-B BPB | Relative change |
| --- | ---: | ---: | ---: |
| Code | 0.686505 | 0.739298 | −7.14% |
| English | 0.845356 | 0.894396 | −5.48% |
| Japanese / Korean | 0.953428 | 0.964779 | −1.18% |
| Math | 0.664294 | 0.701121 | −5.25% |
| Multilingual | 1.110032 | 1.224230 | −9.33% |
| zh-TW / zh | 1.126968 | 1.137099 | −0.89% |

六桶全部通過，且沒有任何 bucket 相對 Role-B 變差。這證明 frozen likelihood retention，不代表所有 downstream benchmark、factuality 或 safety behavior 都已完整驗證。

## 為什麼 release gate 是 5/7，而不是要求 7/7？

較早的 strict research gate 曾要求七類全部通過、每個 evidence depth 都為正、strict-8K operator cells 6/6。這把兩個不同問題混在一起：

- 這個 1.1B base model 是否已具備實用且可重現的 1M evidence-use capability？
- 它是否已具備任意 task、任意位置、任意表面的 universal robustness？

最終 calibrated gate 在 terminal run 前凍結，保留三個不可省略的要求：exact-1M 全部 finite、至少 5/7 task-level statistical effects 成立、6/6 base-model retention；同時不把 free-generation、7/7、每個 depth、strict-8K 6/6 升格為 release blocker。

這不是把失敗藏起來。Aggregation 仍明確標為未證明，strict-8K operator diagnostic 仍只有 3/6。Calibrated stable 1M 的意思是「範圍明確的能力成立」，不是「所有長上下文問題已完成」。

## 哪些捷徑沒有奏效？

整個計畫最重要的進展，有不少來自正式保留的負面結果：

- **只改 config：** 能表達 1M positions，不代表權重會使用它們。
- **只看 training health：** 1M forward/backward 沒有 OOM 或 NaN，只能證明 runtime。
- **只看 BPB：** 128K 階段曾出現 BPB 改善，但 incremental distant-context effect 沒有同步成立。
- **只做 NIAH：** literal retrieval 可以通過，aggregation／state update 仍可能失敗。
- **只增加相似 synthetic templates：** 模型可能學會 surface shortcut，而不是可轉移的 evidence computation。
- **pure-KDA conversion：** 原始能力 retention 不足，因此沒有進入 release lineage。
- **把 strict diagnostics 全部變成 release blocker：** 對 1.1B base model 會把 scoped capability 與 universal robustness 混為一談。

每個 hypothesis 都使用一次 frozen evaluation；同一 fixture 失敗後不以重跑、換 seed 或事後降低門檻硬轉成 PASS。Pre-forward validator bugs 因為 0 optimizer update，不計入訓練 budget。

## 最終發布

正式 checkpoint：

```text
final-global-barbet-iter7008-stable-1m-v11
```

Hugging Face repository：[`OpenFormosa/barbet-1b-base`](https://huggingface.co/OpenFormosa/barbet-1b-base)

可核對 identity：

| Artifact | Identity |
| --- | --- |
| Hugging Face revision | `bd8de3ec404752d61da9df78ab2d1f59928f7f44` |
| Checkpoint tree SHA256 | `0380aee774af7f1e81c8e4b99d016171b07490999a117f66653d4c520b580cf6` |
| `model.safetensors` SHA256 | `05376dde654c9beb8154e1e997d0765ce4e236e51d4de99f38ad2e20fd730000` |
| `config.json` SHA256 | `968e293a32e225a993e9a0e503ed4f41f0d3c69e7d7b03c85078497935b48d91` |

目前 repository 內的 `configs/barbet_1b/` 與 `configs/barbet_1b_1m/` 保留為原始 R2／legacy extrapolation presets，方便重現舊版研究設定；它們不是 final iter7008 的 release config。載入正式模型時，請直接使用 Hugging Face repository 內隨權重發布的 `config.json`。

## 能力邊界

Barbet iter7008 可以合理宣稱：

> 一個約 1.119B parameters 的 causal base model，具有原生 1,048,576-token runtime，並在六類 frozen exact-1M base-LM likelihood tasks 上顯示可重現的遠距 evidence-use effect，同時保留六類原始 base-model BPB。

它不能被解讀為：

- 能完整理解任意一百萬 token 文件；
- aggregation 已可靠；
- 每個 evidence depth 都穩定；
- teacher-forced likelihood 等同可靠 free generation；
- 已具備 instruction-following、聊天或安全對齊；
- 已證明 native 2M；
- 已達到大型 frontier foundation model 的整體能力。

## 結語

把 Barbet 做到 1M，最困難的部分不是把整數從 262,144 改成 1,048,576，而是建立一條能被逐步否證的證據鏈：

```text
原始函數正確
→ 長序列能訓練
→ 位置範圍原生化
→ 遠距 token 有直接路徑
→ coherent evidence 產生可歸因效果
→ 原始 base-model 能力沒有遺忘
```

最終得到的不是「萬能 1M」，而是一個邊界清楚、可重現、知道自己仍缺什麼的 stable 1M base model。對研究模型而言，這比一個只有漂亮 context-window 數字、卻無法分辨 runtime 與 capability 的版本更有價值。
