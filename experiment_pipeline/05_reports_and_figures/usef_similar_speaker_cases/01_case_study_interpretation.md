# Case Study 解读与第二 Verifier 复核

## 这批 case study 在看什么

这批 case study 不是为了证明 USEF 整体强不强，而是为了回答：

```text
USEF-TFGridNet 在 similar-speaker 条件下出现的局部 mismatch，
到底是真正的 target/interferer swap，
还是 speaker verifier 对相似说话人或模型输出不稳定？
```

我们从 taxonomy v2 中每类挑了 5 个样本，共 25 个。

本地位置：

```text
05_reports_and_figures/usef_similar_speaker_cases/
```

## 第一层：ECAPA case study 结果

按 ECAPA 指标看，五类 case 的平均情况如下。

| Case type | SI-SNRi | ECAPA mismatch rate | true swap rate | identity ambiguity rate | local SI-SDR gap | 初步含义 |
|---|---:|---:|---:|---:|---:|---|
| `waveform_global_failure` | -84.17 | 0.754 | 0.754 | 0.000 | -59.35 | 波形层面真的失败 |
| `true_local_swap` | -53.36 | 1.000 | 1.000 | 0.000 | -58.20 | 波形和身份都偏向 interferer |
| `local_identity_ambiguity` | 21.36 | 1.000 | 0.000 | 1.000 | 64.04 | SI-SDR 很好，但 ECAPA 认为身份错 |
| `global_identity_disagreement` | 22.58 | 1.000 | 0.000 | 1.000 | 56.04 | SI-SDR 很好，但 ECAPA 全局不同意 |
| `recovery_candidate` | 20.65 | 0.417 | 0.000 | 0.417 | 52.24 | 局部短时身份不稳定候选 |

这个结果把样本自然分成三组：

1. **真实波形失败组**
   ```text
   waveform_global_failure
   true_local_swap
   ```
   这两类的 SI-SNRi 很低，local SI-SDR gap 也为负。它们更像模型真的提错了或整句失败。

2. **高 SI-SNR 但 ECAPA 反对组**
   ```text
   local_identity_ambiguity
   global_identity_disagreement
   ```
   这两类非常有趣：SI-SNRi 在 20 dB 以上，local SI-SDR gap 高达 56-64 dB，但 ECAPA mismatch rate 是 1.0。

3. **局部恢复候选**
   ```text
   recovery_candidate
   ```
   这类 SI-SNRi 很高，ECAPA mismatch rate 约 0.42，可能是短时局部身份不稳定，也可能是 verifier 噪声。

## 第二层：SpeechBrain x-vector 复核

为了避免只相信 ECAPA，我们使用 SpeechBrain 的独立 speaker verifier 做复核：

```text
source: speechbrain/spkrec-xvect-voxceleb
sample rate: 16 kHz
chunk: 1s
```

复核时用了两种 target 原型：

```text
target enrollment cue
target reference source
```

这样可以区分：

```text
cue 本身不稳定
vs.
target 源身份整体不稳定
```

## 第二 verifier 结论

| Case type | ECAPA mismatch | SpeechBrain mismatch using enrollment | SpeechBrain mismatch using target ref | 结论 |
|---|---:|---:|---:|---|
| `waveform_global_failure` | 0.754 | 0.967 | 1.000 | 支持 ECAPA，确实失败 |
| `true_local_swap` | 1.000 | 0.933 | 0.933 | 支持 ECAPA，确实偏向 interferer |
| `local_identity_ambiguity` | 1.000 | 0.080 | 0.040 | 不支持 ECAPA，疑似 ECAPA artifact |
| `global_identity_disagreement` | 1.000 | 0.000 | 0.040 | 不支持 ECAPA，疑似 ECAPA artifact |
| `recovery_candidate` | 0.417 | 0.345 | 0.197 | 部分支持，但强度下降 |

最关键的发现：

```text
SpeechBrain verifier 支持 waveform failure / true swap，
但强烈否定 ECAPA 对 local_identity_ambiguity 和 global_identity_disagreement 的判断。
```

这说明之前看到的大量 `local_identity_ambiguity` 不能直接解释成“模型真的局部跟错人”。更稳妥的解释是：

```text
ECAPA 在某些高 SI-SNR 输出上出现 verifier-level disagreement；
这可能来自相似说话人边界、短 chunk embedding 不稳定、
或者 ECAPA 与 USEF 输出语音的域不匹配。
```

## 对研究问题的影响

这个复核不是坏消息，反而让问题更清楚。

目前最稳的结论是：

```text
USEF-TFGridNet 在 similar-speaker full test 上整体很强；
确实存在少量真正 waveform-level swap / failure；
但大规模 identity ambiguity 主要由 ECAPA 单 verifier 触发，
需要多 verifier 或人工听感确认后才能写成模型 mismatch。
```

因此后续不能直接说：

```text
USEF 有 2935 / 6000 条 local identity ambiguity。
```

更准确的写法应该是：

```text
ECAPA 检测到大量 verifier-level identity disagreement；
但 SpeechBrain x-vector 复核显示，这些 disagreement 大多不是稳定的 speaker-verifier 共识。
```

## 现在最可信的 case 类型

可以优先用于汇报的强证据：

```text
waveform_global_failure
true_local_swap
```

需要谨慎讨论的探索性证据：

```text
local_identity_ambiguity
global_identity_disagreement
recovery_candidate
```

其中 `global_identity_disagreement` 更适合作为 verifier 稳定性问题，而不是模型失败问题。

## 下一步建议

1. 人工听 `true_local_swap` 和 `waveform_global_failure`，确认是否真的听起来像 interferer。
2. 对 `local_identity_ambiguity` 做人工听感，判断输出是否明显是 target。
3. 若人工听感支持 SpeechBrain，则把 ECAPA 大规模 ambiguity 改写成“single-verifier disagreement”。
4. similar content 阶段必须引入 content verifier，不能复用 speaker verifier 单独下结论。
