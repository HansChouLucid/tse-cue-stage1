# Pipeline Status

更新时间：2026-05-24

## 当前阶段

项目已经进入 controlled single-cue comparison 阶段。

已完成：

```text
01 hard-condition 数据构造
02 chunk-level verifier sanity
03 官方 pretrained USEF-TFGridNet diagnostics
05 USEF condition-aware failure taxonomy
04 normal train-100 single-cue training: TFMap, Context, USEF
04 100-sample sanity diagnostics: TFMap, Context, USEF
04 full ECAPA local mismatch diagnostics: TFMap, Context, USEF
04 full similar-content v2 TTS ECAPA diagnostics: TFMap, Context, USEF
04 full similar-content v2 TTS SSL content diagnostics: TFMap, Context, USEF
```

仍未完成：

```text
stronger retrieval-style content verifier calibration
v2 TTS 的 ASR/token-level readable evidence
v2 TTS 的 case-level attribution：speaker drift / content drift / separation failure
```

## 阶段总览

| Stage | 目的 | 状态 |
|---|---|---|
| 01 Data preparation | 构造 similar-speaker、similar-content v1、TTS similar-content v2 数据；避免 enrollment leakage | done |
| 02 Verifier sanity | 验证 chunk-level verifier 在 clean chunks 上是否可用 | ECAPA 和第一版 SSL sanity 已完成 |
| 03 USEF-only diagnostic | 在 hard conditions 上运行官方 pretrained USEF-TFGridNet，并计算 local metrics | v1/v2 已完成 |
| 04 Controlled single-cue comparison | 在同一 WeSep BSRNN 设置下训练 TFMap/Context/USEF，并诊断 local mismatch | train-100 checkpoints、full speaker-side diagnostics、v2 TTS SSL content diagnostics 已完成 |
| 05 Reports and taxonomy | 汇总 USEF cases、second verifier、content verifier、failure attribution | USEF 主线已完成 |

## 官方 Pretrained USEF 结果

| Condition | Utterances | SI-SNRi mean | SI-SNRi median | SI-SNRi < 0 |
|---|---:|---:|---:|---:|
| similar-speaker | 6000 | 15.86 dB | 18.49 dB | 323 |
| similar-content v1 | 5989 | 16.84 dB | 18.81 dB | 215 |
| similar-content v2 TTS | 5984 | 13.18 dB | 15.81 dB | 343 |
| normal/easy reference | 6000 | 15.90 dB | 18.69 dB | 310 |

解释：

```text
pretrained USEF 的最强证据是 local/tail-focused，而不是只看平均 SI-SNRi 下降。
similar-content v2 TTS 显示最清楚的全局性能下降。
similar-speaker 和 similar-content v1 更适合从 tail failure 和 local mismatch attribution 角度解释。
```

## Controlled Single-Cue Training

三个 WeSep BSRNN single-cue checkpoints 都已在 normal/basic LibriMix-style train-100 上完成训练。

| Cue | Epochs | Final val loss | Checkpoint |
|---|---:|---:|---|
| TFMap-only | 5 | -4.208 | `normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_5.pt` |
| Context-only | 5 | -4.734 | `normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_5.pt` |
| USEF-only | 5 | -8.348 | `normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_5.pt` |

注意：这些是 controlled train-100 checkpoints，不是官方强 pretrained models。

## 100-Sample Controlled Sanity

100-sample diagnostic 说明当前 pipeline 可以运行，指标不是退化的常数，能够区分不同 cue。

| Condition | Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap |
|---|---|---:|---:|---:|
| similar-speaker | TFMap | 0.077 | 0.342 | 8.901 |
| similar-speaker | Context | 0.048 | 0.390 | 4.963 |
| similar-speaker | USEF | 0.164 | 0.167 | 23.348 |
| similar-content v1 | TFMap | 0.213 | 0.208 | 16.247 |
| similar-content v1 | Context | 0.218 | 0.187 | 17.381 |
| similar-content v1 | USEF | 0.294 | 0.084 | 27.466 |

初步解读：

```text
在 100-sample sanity 中，USEF-only 的局部身份更稳定。
TFMap/Context 的 local speaker mismatch rate 更高，尤其是在 similar-speaker 条件下。
这只是 sanity，不是最终结论；最终判断需要 full diagnostics。
```

## Full Controlled Single-Cue Diagnostic

full ECAPA speaker-side diagnostics 已完成。

| Condition | Cue | 1s speaker gap mean | 1s mismatch rate | 1s local SI-SDR gap | 2s mismatch rate |
|---|---|---:|---:|---:|---:|
| similar-speaker | TFMap | 0.078 | 0.341 | 8.575 | 0.302 |
| similar-speaker | Context | 0.046 | 0.404 | 4.776 | 0.379 |
| similar-speaker | USEF | 0.174 | 0.142 | 24.034 | 0.093 |
| similar-content v1 | TFMap | 0.212 | 0.204 | 15.439 | 0.163 |
| similar-content v1 | Context | 0.212 | 0.206 | 15.979 | 0.171 |
| similar-content v1 | USEF | 0.303 | 0.083 | 27.348 | 0.056 |
| similar-content v2 TTS | TFMap | 0.220 | 0.172 | 15.841 | 0.136 |
| similar-content v2 TTS | Context | 0.218 | 0.178 | 16.841 | 0.142 |
| similar-content v2 TTS | USEF | 0.257 | 0.121 | 21.887 | 0.091 |

当前解读：

```text
在 controlled train-100 / BSRNN 设置下，USEF-only 的 local speaker identity 最稳定。
TFMap-only 和 Context-only 的 local speaker mismatch 明显更高，similar-speaker 条件下尤其明显。
similar-content v1/v2 的 speaker-side mismatch 弱于 similar-speaker，但 v2 TTS 更接近 content stress test。
v2 中 USEF 仍然最稳定，TFMap 和 Context 接近；这说明 v2 没有把 speaker-side diagnostic 完全打乱。
v2 的核心问题是 content attribution；SSL content diagnostic 已完成，后续需要 ASR/token-level 和 case-level attribution。
```

## Similar-Content v2 TTS SSL Content Diagnostic

该诊断比较输出 chunk 与 target/interferer reference 的 wav2vec2-base content embedding：

```text
content_gap = sim(output, target_content) - sim(output, interferer_content)
content_mismatch = content_gap < 0
```

| Cue | SI-SNRi mean | SI-SNRi < 0 | content gap mean | content mismatch rate | content mismatch p90 | any content mismatch | joint content+waveform rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| TFMap | 4.593 | 729 / 5984 | 0.084 | 0.283 | 0.778 | 0.628 | 0.103 |
| Context | 4.546 | 955 / 5984 | 0.082 | 0.298 | 0.800 | 0.617 | 0.122 |
| USEF | 8.124 | 310 / 5984 | 0.191 | 0.083 | 0.250 | 0.272 | 0.036 |

解读：

```text
USEF-only 在 content attribution 上也明显更稳定。
TFMap/Context 的 content mismatch rate 约为 USEF 的 3.4-3.6 倍。
低 SI-SNRi 样本中三种 cue 的 content mismatch mean 都约为 0.70，说明极端失败与 content drift 强相关。
```

## Verifier 状态

chunk-level verifier validation 是主逻辑的一部分。

| Verifier / Metric | 当前证据 | 状态 |
|---|---|---|
| ECAPA speaker verifier on clean similar-speaker chunks | 1s/2s chunk 上 AUC 和 accuracy 较高 | 可作为 first-pass speaker evidence |
| ECAPA on separated output chunks | 有用，但不能单独作为强证据 | 需要结合 waveform gap 和 second verifier |
| SpeechBrain x-vector | 可以减少 ECAPA-only false positives | 可作为 second speaker verifier |
| local waveform gap | 将 verifier 结果锚定到 target/interferer waveform | strong speaker mismatch claim 必需 |
| SSL content gap | 能解释 similar-content failures；v2 TTS 上 TFMap/Context/USEF full diagnostics 已完成 | 有用，但还需要 ASR/token-level readable evidence 和 stronger retrieval calibration |

## 当前阅读路径

1. `00_overview/01_local_mismatch_definition.md`
2. `01_data_preparation/01_dataset_inventory.md`
3. `02_pilot_verifier_sanity/README.md`
4. `03_pretrained_usef_diagnostic/README.md`
5. `04_controlled_single_cue_comparison/README.md`
6. `05_reports_and_figures/condition_aware_failure_taxonomy/README.md`

## 下一步优先级

1. 对 v2 的高 mismatch / 极端失败样本做 case-level attribution，区分 speaker drift、content drift 和普通 separation failure。
2. 增加 ASR/token-level readable evidence，让 content mismatch 结论更容易解释。
3. 将 full controlled single-cue 结果进一步整理成论文表格和图。
