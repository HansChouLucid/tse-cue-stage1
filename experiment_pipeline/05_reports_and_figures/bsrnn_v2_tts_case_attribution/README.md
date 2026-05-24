# BSRNN v2 TTS Case-Level Attribution

本报告补齐 controlled BSRNN single-cue 实验的细粒度解释。对象是 `similar-content v2 TTS`，模型为同一 WeSep BSRNN backbone 下的 `TFMap-only`、`Context-only`、`USEF-only`。

远端结果目录：

```text
/data/tse_cue_project/experiments/normal_basic_single_cue_reports/bsrnn_v2_tts_case_attribution/
```

## Attribution 规则

每个 utterance 同时合并三类证据：

| Evidence | 来源 | 作用 |
|---|---|---|
| `sisnr_i` | inference summary | 判断整体分离成功或失败 |
| `content_mismatch_rate` | SSL content diagnostic | 判断输出内容是否更接近 interferer reference |
| `joint_content_waveform_rate` | SSL content + waveform gap | 判断 content drift 是否也被 waveform preference 支持 |
| `speaker_mismatch_rate_1s` | ECAPA speaker-side diagnostic | 判断是否发生 speaker identity drift |
| `local_sisdr_gap_mean_1s` | local waveform diagnostic | 判断输出局部更接近 target 还是 interferer waveform |

主要 attribution label：

| Label | 含义 |
|---|---|
| `confirmed_content_drift` | content mismatch 有 waveform 支持，是 v2 最核心的失败类型 |
| `mixed_content_and_speaker_drift` | content drift 和 speaker drift 同时出现 |
| `speaker_drift_or_identity_instability` | 主要表现为 speaker-side drift |
| `content_embedding_only_without_waveform_support` | SSL content 偏移明显，但 waveform 不支持，证据较弱 |
| `ordinary_separation_failure_or_unattributed` | 低 SI-SNRi，但当前 mismatch 指标未能解释 |
| `stable_success` | content 与 speaker 都较稳定 |
| `ambiguous_or_mild_local_error` | 有轻微局部异常，但不足以归入强失败类型 |

## 全量 Case Attribution

| Cue | SI-SNRi mean | content mismatch mean | joint content+waveform mean | speaker mismatch mean | confirmed content drift | mixed drift | stable success |
|---|---:|---:|---:|---:|---:|---:|---:|
| TFMap-only | 4.597 | 0.283 | 0.103 | 0.110 | 879 | 486 | 1858 |
| Context-only | 4.548 | 0.298 | 0.122 | 0.114 | 1015 | 533 | 1920 |
| USEF-only | 8.124 | 0.083 | 0.036 | 0.142 | 259 | 83 | 2277 |

解读：

```text
Context-only 的 content drift 最严重，TFMap-only 次之，USEF-only 明显更稳定。
USEF-only 的 overall SI-SNRi 更高，confirmed content drift 和 mixed drift 数量都显著更少。
USEF-only 的 speaker_mismatch_mean 不低，说明它不是在所有局部指标上都完美；但它更少进入 content attribution collapse。
```

## 极端失败样本

| Cue | SI-SNRi < 0 | low-SI-SNRi content mismatch mean | confirmed content drift | mixed drift | unattributed failure |
|---|---:|---:|---:|---:|---:|
| TFMap-only | 726 | 0.697 | 420 | 230 | 63 |
| Context-only | 954 | 0.706 | 574 | 301 | 66 |
| USEF-only | 310 | 0.718 | 214 | 58 | 22 |

解读：

```text
一旦模型进入极端失败，三种 cue 的 content mismatch 都很高，平均约 0.70。
差异不在于“失败后是否 content drift”，而在于“进入失败状态的频率”。
Context 进入极端失败最多，TFMap 次之，USEF 最少。
```

## ASR / Token-Level Evidence

对 top cases 跑了 wav2vec2 ASR。注意：v2 TTS 中 synthetic interferer 使用 target text 合成，因此 ASR 主要用于检验输出是否保留 target text、是否可读；它不能单独区分 target text 与 interferer text。区分 target/interferer 仍需要 SSL content + waveform/speaker 证据。

远端输出：

```text
asr_top_cases/asr_topcase_rows.csv
asr_top_cases/asr_topcase_summary.json
```

| Cue / Group | n | WER mean | WER median | token F1 mean | token F1 median |
|---|---:|---:|---:|---:|---:|
| TFMap content drift | 10 | 1.031 | 1.000 | 0.047 | 0.000 |
| Context content drift | 10 | 1.018 | 1.000 | 0.028 | 0.000 |
| USEF content drift | 10 | 1.023 | 1.000 | 0.099 | 0.093 |
| TFMap extreme failure | 10 | 1.113 | 1.000 | 0.054 | 0.028 |
| Context extreme failure | 10 | 1.009 | 1.000 | 0.037 | 0.000 |
| USEF extreme failure | 10 | 1.002 | 1.000 | 0.082 | 0.069 |
| TFMap cross-cue recovery | 20 | 0.940 | 1.000 | 0.146 | 0.101 |
| Context cross-cue recovery | 20 | 0.962 | 0.957 | 0.115 | 0.119 |
| USEF cross-cue recovery | 20 | 0.679 | 0.717 | 0.426 | 0.433 |

解读：

```text
content drift / extreme failure 的输出文本基本不可读或严重偏离 target，WER 接近或超过 1，token F1 很低。
在 cross-cue recovery candidates 中，同一批样本下 USEF 的 WER 明显低于 TFMap/Context，token F1 明显更高。
这说明 USEF 不只是 SSL embedding 分数更好，在可读文本层面也更能保留 target content。
```

## 与 Pretrained USEF Case Study 的关系

之前强 pretrained USEF 的 case study 主要回答：

```text
强模型在 hard condition 下仍会出现局部 mismatch；
这些 mismatch 可以用 ECAPA / SpeechBrain / waveform / SSL content 等多证据链解释。
```

本 BSRNN controlled case attribution 回答：

```text
在同一 backbone、同一训练集、同一诊断流程下，不同 cue 的 mismatch 行为不同；
USEF cue 在 speaker/content 两条线上都更稳定，TFMap/Context 更容易出现 content drift。
```

两条证据链互补：

| Evidence line | 作用 |
|---|---|
| Pretrained USEF case study | 证明问题在强模型中真实存在，不只是弱 baseline artifact |
| Controlled BSRNN cue comparison | 证明不同 cue 机制会系统性改变 mismatch 行为 |
| v2 TTS SSL + ASR evidence | 证明 content attribution mismatch 可以被量化，并可被文本层面佐证 |

## 当前边界

```text
这些 BSRNN checkpoints 仍然是 train-100 / 5 epoch 的 controlled models，不是强 pretrained baseline。
因此它们适合支持 cue-mechanism comparison，不适合单独作为最终 SOTA 性能比较。
下一步如果进入 solution，需要保留 pretrained USEF 作为强模型参照，同时把 BSRNN 作为可控实验台。
```
