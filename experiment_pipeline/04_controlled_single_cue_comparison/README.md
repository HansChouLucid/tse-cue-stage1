# Controlled Single-Cue Comparison

本阶段比较三种 fine-grained cue 机制：TFMap、Context、USEF。

比较方式是：在同一个 WeSep BSRNN backbone、同一套 normal/basic LibriMix train-100 训练设置下，训练三个 single-cue checkpoints，然后观察它们在 hard conditions 下的 local target/interferer mismatch 行为是否不同。

这个阶段的目标不是超过官方 USEF-TFGridNet pretrained model，而是得到可控、可比较的 single-cue checkpoints，用于分析 cue 机制本身。

## 已完成训练

训练数据：内部 normal/basic LibriMix-style train-100。

- Source metadata：Libri2Mix train-clean-100 metadata。
- Source audio：LibriSpeech train-clean-100。
- Mixtures：13,900 条 two-speaker mixtures。
- Materialized wavs：41,700 个文件，包含 `mix/s1/s2`。
- Enrollment policy：same-speaker enrollment，尽量避免使用当前 target utterance。
- Backbone：WeSep `TSE_BSRNN_SPK`。
- Epochs：5。

| Cue | Status | Batch / AMP | Epoch 5 train loss | Epoch 5 val loss | Checkpoint |
|---|---|---:|---:|---:|---|
| TFMap-only | done | bs16 / AMP | -4.989 | -4.208 | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_5.pt` |
| Context-only | done | bs4 / no AMP | -6.049 | -4.734 | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_5.pt` |
| USEF-only | done | bs8 / AMP | -9.028 | -8.348 | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_5.pt` |

补充：USEF-only 最初使用 bs16 时 OOM，后来降到 bs8 后完成训练。

## 已完成 Sanity Diagnostic

Diagnostic script：`tools/run_model_output_mismatch_pilot.py`。

该脚本会直接运行模型推理，并在模型输出上计算 chunk-level local mismatch metrics。

主要指标：

| Metric | 含义 |
|---|---|
| `speaker_gap` | 输出 chunk 与 target speaker prototype 的 ECAPA 相似度减去与 interferer prototype 的相似度。越高越好。 |
| `mismatch_rate` | 有效 local chunks 中 `speaker_gap < 0` 的比例。越高表示越多 local speaker identity drift。 |
| `local_sisdr_gap` | 输出 chunk 更接近 target waveform 还是 interferer waveform。越高表示更偏 target。 |

Sanity split：每个 hard test set 的前 100 条样本。每个 mixture 会生成两个 target tasks，因此报告中的 utterance count 约为 200。

### Similar-Speaker Sanity

| Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 2s mismatch rate |
|---|---:|---:|---:|---:|
| TFMap-only | 0.077 | 0.342 | 8.901 | 0.313 |
| Context-only | 0.048 | 0.390 | 4.963 | 0.366 |
| USEF-only | 0.164 | 0.167 | 23.348 | 0.114 |

初步解读：在这个小 sanity set 上，USEF-only 的 local speaker identity 更稳定；TFMap 和 Context 在 similar-speaker stress 下出现更高的 local mismatch rate。

### Similar-Content v1 Sanity

| Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 2s mismatch rate |
|---|---:|---:|---:|---:|
| TFMap-only | 0.213 | 0.208 | 16.247 | 0.172 |
| Context-only | 0.218 | 0.187 | 17.381 | 0.170 |
| USEF-only | 0.294 | 0.084 | 27.466 | 0.064 |

初步解读：similar-content v1 中 USEF-only 的 local speaker mismatch 也更低。但这里使用的是 speaker diagnostic，不是完整 content diagnostic；similar-content 的最终解释还需要 SSL/content metrics。

## Full Diagnostic 结果

full ECAPA local mismatch diagnostics 已在三个 full test sets 上完成：

| Condition | Samples / target tasks | Cue |
|---|---:|---|
| similar-speaker test | 6000 mixtures / 12000 target tasks | TFMap, Context, USEF |
| similar-content v1 test | 5989 mixtures / 11978 target tasks | TFMap, Context, USEF |
| similar-content v2 TTS test | 5984 mixtures / 11968 target tasks | TFMap, Context, USEF |

输出根目录：

```text
/data/tse_cue_project/experiments/normal_basic_single_cue_diagnostic/
```

### Similar-Speaker Full

| Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 1s high-mismatch tasks (`rate >= 0.5`) | 1s zero-mismatch tasks | 2s mismatch rate |
|---|---:|---:|---:|---:|---:|---:|
| TFMap-only | 0.078 | 0.341 | 8.575 | 3677 / 12000 | 2327 / 12000 | 0.302 |
| Context-only | 0.046 | 0.404 | 4.776 | 4710 / 12000 | 1669 / 12000 | 0.379 |
| USEF-only | 0.174 | 0.142 | 24.034 | 1078 / 12000 | 6397 / 12000 | 0.093 |

解读：

```text
similar-speaker 是最能暴露 speaker identity mismatch 的条件。
Context-only 的 local mismatch 最严重，TFMap-only 次之，USEF-only 明显更稳定。
USEF-only 不只是 mismatch rate 更低，local SI-SDR gap 也显著更高，说明输出在 waveform 层面更偏 target。
```

### Similar-Content v1 Full

| Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 1s high-mismatch tasks (`rate >= 0.5`) | 1s zero-mismatch tasks | 2s mismatch rate |
|---|---:|---:|---:|---:|---:|---:|
| TFMap-only | 0.212 | 0.204 | 15.439 | 1922 / 11978 | 5105 / 11978 | 0.163 |
| Context-only | 0.212 | 0.206 | 15.979 | 2031 / 11978 | 5334 / 11978 | 0.171 |
| USEF-only | 0.303 | 0.083 | 27.348 | 633 / 11978 | 8466 / 11978 | 0.056 |

解读：

```text
similar-content v1 上，speaker-side mismatch 仍然存在，但弱于 similar-speaker。
TFMap-only 和 Context-only 的 speaker mismatch 水平非常接近。
USEF-only 仍然明显更稳定，且 2s p90 mismatch rate 为 0，说明大多数样本没有持续的 speaker identity drift。
```

### Similar-Content v2 TTS Full

v2 TTS 数据把 target transcript 通过 TTS 方式注入到 interferer voice 中，因此它比 v1 更接近 content stress test。这里的表格仍然是 ECAPA speaker-side diagnostic，不是最终 content diagnostic。

| Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 1s high-mismatch tasks (`rate >= 0.5`) | 1s zero-mismatch tasks | 2s mismatch rate |
|---|---:|---:|---:|---:|---:|---:|
| TFMap-only | 0.220 | 0.172 | 15.841 | 1710 / 11960 | 6421 / 11960 | 0.136 |
| Context-only | 0.218 | 0.178 | 16.841 | 1854 / 11960 | 6537 / 11960 | 0.142 |
| USEF-only | 0.257 | 0.121 | 21.887 | 1201 / 11960 | 7983 / 11960 | 0.091 |

解读：

```text
v2 TTS 中，三种 cue 的 speaker-side mismatch rate 都低于 similar-speaker，但高于或接近各自在 similar-content v1 中的相邻水平。
USEF-only 仍然最稳定；TFMap-only 和 Context-only 非常接近，Context 的 high-mismatch tasks 略多。
v2 的 local SI-SDR gap 比 v1 中 USEF-only 的 gap 明显低，说明 v2 对模型输出质量确实更难。
但是 ECAPA speaker-side diagnostic 只能说明输出是否发生 speaker identity drift，不能完整判断 content attribution drift。
因此 v2 的下一步重点应是 content-specific SSL/ASR-style diagnostic，而不是只根据 ECAPA 下结论。
```

### Similar-Content v2 TTS SSL Content Diagnostic

已完成 TFMap/Context/USEF 在 v2 TTS 上的完整 SSL content diagnostic。该诊断使用 wav2vec2-base 中间层作为 content embedding，比较每个 1s chunk 的输出与 target reference、interferer reference 的 content 相似度：

```text
content_gap = sim(output, target_content) - sim(output, interferer_content)
content_mismatch = content_gap < 0
joint_content_waveform = content_mismatch 且 waveform 更接近 interferer
```

输出路径：

```text
/data/tse_cue_project/experiments/normal_basic_single_cue_content_diagnostic/similar_content_v2_tts/
```

| Cue | SI-SNRi mean | SI-SNRi < 0 | content gap mean | content mismatch rate | content mismatch p90 | any content mismatch | joint content+waveform rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| TFMap-only | 4.593 | 729 / 5984 | 0.084 | 0.283 | 0.778 | 0.628 | 0.103 |
| Context-only | 4.546 | 955 / 5984 | 0.082 | 0.298 | 0.800 | 0.617 | 0.122 |
| USEF-only | 8.124 | 310 / 5984 | 0.191 | 0.083 | 0.250 | 0.272 | 0.036 |

低 SI-SNRi 样本中的 content mismatch：

| Cue | low SI-SNRi samples | low SI-SNRi content mismatch mean | good SI-SNRi samples | good SI-SNRi content mismatch mean |
|---|---:|---:|---:|---:|
| TFMap-only | 726 | 0.697 | 2 | 0.000 |
| Context-only | 954 | 0.706 | 3 | 0.000 |
| USEF-only | 310 | 0.718 | 21 | 0.018 |

解读：

```text
1. v2 TTS 的 content attribution stress 被 SSL diagnostic 明确捕捉到了。
2. TFMap-only 和 Context-only 的 content mismatch 明显高于 USEF-only；Context 略差于 TFMap。
3. USEF-only 不仅 speaker-side mismatch 更低，content-side mismatch 也显著更低，说明 USEF 在当前 controlled setting 下同时更稳地保留 speaker identity 和 content attribution。
4. 极端失败样本与 content mismatch 强相关。三种 cue 中，low SI-SNRi 样本的 content mismatch mean 都在 0.70 左右，说明当模型整体崩溃时，输出往往大面积偏向 interferer content。
5. good SI-SNRi 样本几乎没有 content mismatch，说明 SSL content metric 和传统 SI-SNRi 在 tail failure 上有一致性；但 SSL metric 还能定位局部 mismatch，而不只是给出整句分数。
```

### 三组 Full Diagnostic 横向比较

| Condition | TFMap 1s mismatch | Context 1s mismatch | USEF 1s mismatch | 主要现象 |
|---|---:|---:|---:|---|
| similar-speaker | 0.341 | 0.404 | 0.142 | 最强 speaker identity stress；Context 最不稳定，USEF 明显更稳 |
| similar-content v1 | 0.204 | 0.206 | 0.083 | speaker drift 减弱；TFMap/Context 接近，USEF 仍最稳 |
| similar-content v2 TTS | 0.172 | 0.178 | 0.121 | content stress 更强，但 speaker-side mismatch 没有像 similar-speaker 那样爆发 |

横向结论：

```text
1. similar-speaker 是验证 speaker identity mismatch 的主战场。这里 Context/TFMap 明显比 USEF 更容易局部偏向 interferer speaker。
2. similar-content v1/v2 中，ECAPA speaker mismatch 不是主要矛盾；它更像是辅助信号，用来排除或定位 speaker drift。
3. v2 TTS 相比 v1 更难，尤其会降低 USEF 的 local SI-SDR gap；但它没有导致大规模 speaker identity collapse。
4. 因此，v2 的核心研究价值应落在 content attribution mismatch：模型是否在局部跟随 interferer voice 中的 target-like content，或者把 target/interferer 的内容归属混淆。
5. 新增 SSL content diagnostic 后，v2 的 content attribution mismatch 已经从推测变成可量化结果：TFMap/Context 明显更容易发生 content drift，USEF 更稳。
```

## 当前结论

在 controlled train-100 / BSRNN 设置下，三种 fine-grained cue 的局部稳定性排序很清楚：

```text
USEF-only 最稳定  >  TFMap-only  >  Context-only（similar-speaker 下最弱）
```

加入 similar-content v2 TTS 后，结论需要稍微细化：

```text
speaker-side mismatch 排序总体仍支持 USEF-only 最稳定。
similar-speaker 下 Context 明显弱于 TFMap；similar-content v1/v2 下 TFMap 和 Context 更接近。
v2 TTS 的主要价值不是证明 speaker identity drift 更强，而是暴露 content attribution mismatch。
SSL content diagnostic 显示 TFMap/Context 的 content mismatch 明显高于 USEF。
```

这支持两个判断：

1. local mismatch 不是单纯由数据构造或 verifier artifact 造成的。不同 cue 在同一 backbone、同一训练数据下表现出系统性差异。
2. similar-speaker 更适合检验 speaker identity mismatch；similar-content v2 TTS 更适合检验 content attribution mismatch，且需要结合 SSL/ASR-style diagnostic 解释。

## 重要限制

这些结果来自 controlled train-100 BSRNN checkpoints，不是官方强 pretrained checkpoints。因此它们更适合回答“不同 cue 机制是否有不同 local mismatch 行为”，不适合直接作为最终 SOTA 性能比较。

## 解释边界

这些 checkpoints 是 controlled train-100 BSRNN checkpoints，不是官方强 pretrained checkpoints。因此：

- 它们适合在 WeSep 内部做 cue-mechanism comparison。
- 它们不适合作为最终性能模型去和官方 USEF-TFGridNet pretrained model 比强弱。
- 任何关于 fine-grained cue mismatch 的结论都应该基于 full diagnostics，而不是只看 100-sample sanity。
- similar-content，尤其是 v2 TTS，不能只依赖 ECAPA speaker mismatch；当前已经补充 SSL content diagnostic，后续还需要 ASR/token-level 和 case-level attribution 做可读性佐证。
