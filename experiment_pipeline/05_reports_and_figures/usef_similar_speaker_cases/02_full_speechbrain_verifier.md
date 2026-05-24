# Full 6000 SpeechBrain Verifier 复核

## 目的

ECAPA 在 full similar-speaker test 上检测到大量局部 identity disagreement。为了判断这些 disagreement 是否可靠，我们在完整 6000 条 USEF 输出上追加 SpeechBrain x-vector speaker verifier。

模型：

```text
SpeechBrain x-vector
source: speechbrain/spkrec-xvect-voxceleb
sample rate: 16 kHz
chunk: 1s
```

本地结果：

```text
speechbrain_xvector_full/speechbrain_full_summary_1s.json
speechbrain_xvector_full/speechbrain_full_per_utterance_1s.csv
```

远端结果：

```text
/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_speaker_test_full/usef_tfgridnet_wsj0_2mix/speechbrain_xvector_full
```

## 关键定义

单 verifier mismatch：

```text
ECAPA mismatch: speaker_gap_ecapa < 0
SpeechBrain mismatch: speaker_gap_speechbrain < 0
```

multi-verifier mismatch：

```text
ECAPA mismatch
and SpeechBrain mismatch
```

multi-verifier true swap：

```text
ECAPA mismatch
and SpeechBrain mismatch
and local_sisdr_gap < 0
```

注意：`local_sisdr_gap < 0` 是强波形证据，因此这个定义是高精度、低召回的。它不能覆盖所有可能的 speaker identity mismatch。

## Chunk-Level 统计

| Metric | Count / N | Rate |
|---|---:|---:|
| ECAPA mismatch | 15189 / 53114 | 28.60% |
| SpeechBrain mismatch using enrollment | 16840 / 53114 | 31.71% |
| SpeechBrain mismatch using target reference | 9124 / 53114 | 17.18% |
| multi-verifier mismatch using enrollment | 5781 / 53114 | 10.88% |
| multi-verifier mismatch using target reference | 3713 / 53114 | 6.99% |
| ECAPA true swap | 1719 / 53114 | 3.24% |
| multi-verifier true swap using enrollment | 1373 / 53114 | 2.59% |
| multi-verifier true swap using target reference | 1289 / 53114 | 2.43% |

## Utterance-Level: 至少一个 chunk

| Metric | Count / 6000 | Rate |
|---|---:|---:|
| ECAPA mismatch | 4760 | 79.33% |
| SpeechBrain mismatch using enrollment | 3627 | 60.45% |
| SpeechBrain mismatch using target reference | 2262 | 37.70% |
| multi-verifier mismatch using enrollment | 2356 | 39.27% |
| multi-verifier mismatch using target reference | 1451 | 24.18% |
| ECAPA true swap | 348 | 5.80% |
| multi-verifier true swap using enrollment | 306 | 5.10% |
| multi-verifier true swap using target reference | 274 | 4.57% |

这个口径非常宽松，因为只要一个 chunk 触发就计入。

## Utterance-Level: 持续性阈值

更稳的口径是看 mismatch rate 是否超过一定比例。

| Threshold | ECAPA mismatch | multi-verifier mismatch target-ref | ECAPA true swap | multi-verifier true swap target-ref |
|---|---:|---:|---:|---:|
| > 0.1 | 4421 / 6000 = 73.68% | 1184 / 6000 = 19.73% | 339 / 6000 = 5.65% | 264 / 6000 = 4.40% |
| > 0.2 | 3146 / 6000 = 52.43% | 654 / 6000 = 10.90% | 290 / 6000 = 4.83% | 222 / 6000 = 3.70% |
| > 0.5 | 1097 / 6000 = 18.28% | 240 / 6000 = 4.00% | 201 / 6000 = 3.35% | 148 / 6000 = 2.47% |
| > 0.75 | 392 / 6000 = 6.53% | 107 / 6000 = 1.78% | 113 / 6000 = 1.88% | 75 / 6000 = 1.25% |

## 解释

1. 单 ECAPA mismatch 很多，但 multi-verifier confirmed mismatch 明显减少。
2. 使用 target reference 作为 SpeechBrain target prototype 时，multi-verifier mismatch 更保守。
3. true swap 被第二 verifier 大体保留：ECAPA true swap 是 5.80%，multi-verifier true swap 是 4.57%。
4. 说明真正波形偏向 interferer 的失败是存在的，但大规模 identity ambiguity 不能只靠 ECAPA 下结论。

## 推荐报告口径

不要写：

```text
USEF 有 79.33% utterance 出现 identity mismatch。
```

应该写：

```text
单 ECAPA verifier 在 79.33% utterance 中检测到至少一个 identity-disagreement chunk；
但经过 SpeechBrain x-vector 复核后，multi-verifier confirmed mismatch 降到 24.18%。
如果进一步要求 mismatch rate > 0.2，则为 10.90%。
真正同时满足 multi-verifier 和 waveform evidence 的 true swap 占 4.57% utterance。
```

## 研究含义

这说明局部 mismatch 检测必须区分三层：

```text
single-verifier disagreement
multi-verifier identity disagreement
waveform-confirmed true swap
```

其中 `true swap` 是最强证据，但可能漏掉一些只发生在身份表征层面的错误。后续 similar-content 研究也需要类似的多证据框架。
