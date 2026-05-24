# Unified Evidence Chain

本文档把两条实验线合并成第一阶段的统一证据链：

```text
强 pretrained USEF 细粒度 case study
+ controlled BSRNN single-cue comparison
+ v2 TTS content attribution diagnostic
```

## 核心结论

当前结果支持：

```text
fine-grained cue TSE 在 hard condition 下存在局部、有限、可诊断的 target/interferer mismatch。
```

mismatch 至少包含两类：

| 类型 | 主要数据 | 主要证据 |
|---|---|---|
| speaker identity mismatch | similar-speaker | ECAPA / SpeechBrain / waveform local SI-SDR gap |
| content attribution mismatch | similar-content v2 TTS | SSL content gap / waveform support / ASR token-level evidence |

## 证据线 A：Pretrained USEF

作用：

```text
证明 mismatch 不是弱 baseline artifact。
即使是较强的官方 pretrained USEF-TFGridNet，在 hard condition 下也会出现 tail failure 和局部 mismatch。
```

已有细粒度结果：

| 分析 | 结论 |
|---|---|
| similar-speaker case study | 存在 true local swap、local identity ambiguity、global identity disagreement 等模式 |
| SpeechBrain second verifier | 复核 ECAPA 的 speaker-side 判断，减少 ECAPA-only false positive |
| condition-aware failure taxonomy | 极端失败大多可以被 mismatch 或 content-driven failure 解释 |
| v2 TTS SSL content diagnostic | v2 TTS 明显造成 content stress，低 SI-SNRi 样本 content mismatch 很高 |

边界：

```text
pretrained USEF 线不能比较 TFMap/Context/USEF 三种 cue，因为只有 USEF 强 checkpoint。
```

## 证据线 B：Controlled BSRNN Single-Cue

作用：

```text
在同一 BSRNN backbone、同一 train-100 数据、同一诊断脚本下比较 TFMap / Context / USEF。
这条线回答 cue mechanism 是否会系统性影响 mismatch。
```

speaker-side full diagnostic：

| Condition | TFMap 1s mismatch | Context 1s mismatch | USEF 1s mismatch |
|---|---:|---:|---:|
| similar-speaker | 0.341 | 0.404 | 0.142 |
| similar-content v1 | 0.204 | 0.206 | 0.083 |
| similar-content v2 TTS | 0.172 | 0.178 | 0.121 |

v2 TTS content-side full diagnostic：

| Cue | SI-SNRi mean | content mismatch rate | joint content+waveform rate |
|---|---:|---:|---:|
| TFMap | 4.593 | 0.283 | 0.103 |
| Context | 4.546 | 0.298 | 0.122 |
| USEF | 8.124 | 0.083 | 0.036 |

结论：

```text
USEF 在 speaker-side 和 content-side 都明显更稳定。
TFMap/Context 更容易出现 content drift，Context 通常略差。
```

边界：

```text
当前 BSRNN checkpoint 是 train-100 / 5 epoch controlled model，不是强 pretrained baseline。
它适合做 cue-mechanism comparison，不适合单独作为 SOTA 性能结论。
```

## 证据线 C：Case-Level Attribution

BSRNN v2 TTS case attribution 把 aggregate 指标落到样本级：

| Cue | confirmed content drift | mixed drift | SI-SNRi < 0 |
|---|---:|---:|---:|
| TFMap | 879 | 486 | 726 |
| Context | 1015 | 533 | 954 |
| USEF | 259 | 83 | 310 |

ASR top-case evidence：

| Group | TFMap token F1 | Context token F1 | USEF token F1 |
|---|---:|---:|---:|
| content drift | 0.047 | 0.028 | 0.099 |
| extreme failure | 0.054 | 0.037 | 0.082 |
| cross-cue recovery | 0.146 | 0.115 | 0.426 |

解读：

```text
top content-drift / extreme-failure cases 的 ASR 文本基本严重偏离 target。
在同一批 cross-cue recovery cases 中，USEF 的文本保真度明显更高。
```

注意：

```text
v2 TTS 的 synthetic interferer 使用 target text 合成，因此 ASR 不能单独区分 target text 与 interferer text。
ASR 的作用是 readable evidence：说明输出是否保留 target text、是否严重崩坏。
target/interferer attribution 仍主要依赖 SSL content gap + waveform/speaker evidence。
```

## 第一阶段是否足够

我认为第一阶段的 mismatch 检验已经基本足够，可以开始进入 solution 设计。

理由：

```text
1. pretrained USEF 证明强模型也会出现 hard-condition local mismatch。
2. controlled BSRNN 证明 cue mechanism 会系统性影响 mismatch。
3. v2 TTS 证明 content attribution mismatch 可以被 SSL content 和 ASR/token evidence 捕捉。
4. case-level attribution 说明 aggregate 指标不是空泛统计，而能落到具体失败模式。
```

## 下一步

建议进入 solution，同时保留必要的 baseline 增强：

| 优先级 | 任务 | 目的 |
|---|---|---|
| P0 | 设计 mismatch-aware solution prototype | 从诊断转向改进 |
| P0 | 在 pretrained USEF 和 controlled BSRNN 上保持同一套 mismatch metrics | 确保 solution 真的降低 mismatch |
| P1 | 如果算力允许，训练 train-360/train-460 BSRNN single-cue baseline | 验证 cue 差异是否随模型增强仍存在 |
| P1 | 补更强 ASR / retrieval-style content verifier | 增强 content mismatch 的可读性和鲁棒性 |

不建议在没有 solution prototype 前无限制扩大 baseline 训练。当前证据已经足够支持进入下一阶段。
