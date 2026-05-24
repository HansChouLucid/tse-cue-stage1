# Confirmed Mismatch 分析

## 目的

本页完成三件事：

1. 分析 confirmed true swap / waveform failure 的原因。
2. 用量化证据替代人工听感，整理强证据 case。
3. 复核 verifier disagreement，判断哪些是真问题，哪些更像 verifier artifact。

## 1. 真失败是否集中

full 6000 上的统计：

| Group | N | Rate | Top-5 speaker-pair share | Top-10 speaker-pair share | Speaker similarity mean |
|---|---:|---:|---:|---:|---:|
| waveform global failure | 323 | 5.38% | 74.61% | 97.21% | 0.456 |
| confirmed true swap, any chunk | 274 | 4.57% | 73.72% | 95.26% | 0.450 |
| confirmed true swap rate > 0.2 | 222 | 3.70% | 80.18% | 97.30% | 0.443 |
| confirmed identity, any chunk | 1451 | 24.18% | 30.32% | 51.83% | 0.442 |
| single ECAPA only | 3309 | 55.15% | 21.70% | 39.20% | 0.451 |

结论：

```text
真失败高度集中在少数 speaker pair；
single-ECAPA-only disagreement 更分散。
```

这说明 confirmed true swap 更像 pair-specific failure，而不是 verifier 随机噪声。

## 2. Top speaker pairs

### Waveform global failure

| Pair | Count | Pair total | Risk within pair |
|---|---:|---:|---:|
| 8555 -> 4970 | 72 | 139 | 51.80% |
| 237 -> 5683 | 54 | 208 | 25.96% |
| 4446 -> 4992 | 49 | 223 | 21.97% |
| 8463 -> 4992 | 41 | 169 | 24.26% |
| 2830 -> 1188 | 25 | 210 | 11.90% |

### Confirmed true swap

| Pair | Count | Pair total | Risk within pair |
|---|---:|---:|---:|
| 8555 -> 4970 | 62 | 139 | 44.60% |
| 237 -> 5683 | 45 | 208 | 21.63% |
| 8463 -> 4992 | 42 | 169 | 24.85% |
| 2830 -> 1188 | 29 | 210 | 13.81% |
| 5683 -> 237 | 24 | 180 | 13.33% |

解释：

```text
8555 -> 4970 是最危险 pair。
它在 full test 中出现 139 次，其中 62 次出现 confirmed true swap。
```

这不是轻微偏差，而是非常明显的 pair-specific failure。

## 3. 是否由 speaker similarity 单独解释

不能。

confirmed true swap 的平均 speaker similarity 是：

```text
confirmed true swap any: 0.450
confirmed true swap rate > 0.2: 0.443
all/single ECAPA only 附近也在 0.45 左右
```

因此：

```text
speaker similarity 是必要背景，但不是充分原因。
```

更可能是：

```text
特定 speaker pair + 局部 overlap + cue/model 域偏差
```

共同触发失败。

## 4. 局部能量和时间位置

量化证据：

| Group | longest true-swap run mean | median first true-swap time | mean interferer-target energy gap on true-swap chunks |
|---|---:|---:|---:|
| waveform global failure | 2.95 chunks | 0.0s | +0.10 dB |
| confirmed true swap, any chunk | 3.57 chunks | 0.0s | +0.20 dB |
| confirmed true swap rate > 0.2 | 4.16 chunks | 0.0s | +0.10 dB |

解释：

1. true swap 往往从开头就出现。
2. true swap 不是单个孤立 chunk，平均连续 3-4 个 1s chunk。
3. interferer 能量只略高于 target，不能单独解释失败。

所以失败更像：

```text
模型在一开始就把目标归属对错了，
并在一段时间内持续提取错误方向。
```

而不是某个瞬时能量峰值导致的偶然错误。

## 5. 强证据 case

case study 中 25 个代表样本复核后：

| Reliability label | Count |
|---|---:|
| confirmed true swap | 10 |
| single ECAPA disagreement | 10 |
| multi-verifier identity disagreement | 1 |
| not confirmed | 4 |

强证据样本主要来自：

```text
waveform_global_failure
true_local_swap
```

它们同时满足：

```text
ECAPA 更像 interferer
SpeechBrain 更像 interferer
local SI-SDR 更接近 interferer
SI-SNRi 很差
```

这类可以作为当前最可靠的 failure case。

## 6. Verifier disagreement 复核

原先最可疑的两类：

```text
local_identity_ambiguity
global_identity_disagreement
```

复核后多数变成：

```text
single ECAPA disagreement
```

也就是：

```text
ECAPA 认为像 interferer，
但 SpeechBrain 不支持，
waveform 也强烈接近 target。
```

因此它们不能直接写成模型 mismatch。

更准确说法：

```text
ECAPA exposes a large amount of single-verifier identity disagreement,
but most high-SI-SNR ambiguity cases are not confirmed by SpeechBrain.
```

## 7. 当前结论

1. USEF-TFGridNet 整体很强。
2. 大规模 ECAPA identity ambiguity 不能直接视为模型错误。
3. multi-verifier + waveform confirmed true swap 真实存在，但占比小。
4. 真失败高度集中在少数 speaker pair。
5. 失败常从开头出现并持续多个 chunk，更像 target attribution 一开始就错，而不是随机局部噪声。

## 8. 下一步

最自然的下一步是：

```text
进入 similar-content diagnostic，
用 content-level verifier 检查内容归属是否也存在类似的局部偏移。
```

同时保留一条支线：

```text
对 top failing speaker pairs 做更细的 pair-specific 分析。
```
