# Clean-source Verifier Sanity Check 结果索引

## 1. 远端脚本

```text
/data/tse_cue_project/code/wesep-real-tse/tools/run_local_mismatch_pilot.py
```

## 2. 远端输出

```text
/data/tse_cue_project/experiments/local_mismatch_pilot
```

已生成：

```text
similar_speaker_test_100
similar_speaker_test_1000
similar_content_test_100
```

## 3. similar speaker test, 1000 samples

实验设置：

```text
chunk = 1.0s, 2.0s
hop = chunk / 2
verifier = ECAPA
prototype utterances = 3
```

结果：

```text
1.0s:
  AUC = 0.9998186482
  accuracy(gap > 0) = 0.9943627188
  target margin mean = 0.3306576914
  target margin p10 = 0.1794299539
  interferer margin mean = 0.3248231309
  interferer margin p10 = 0.1774559838

2.0s:
  AUC = 0.9999920766
  accuracy(gap > 0) = 0.9977157360
  target margin mean = 0.3981363164
  target margin p10 = 0.2642446136
  interferer margin mean = 0.3948117587
  interferer margin p10 = 0.2640495732
```

结论：

ECAPA verifier 在 clean-source 的 `similar_speaker` test 上，1s/2s chunk 均具备很强判别性。

## 4. similar content test, 100 samples

结果：

```text
1.0s:
  AUC = 0.9999934366
  accuracy(gap > 0) = 0.9991460290
  target margin mean = 0.5000208423

2.0s:
  AUC = 1.0000
  accuracy(gap > 0) = 1.0000
  target margin mean = 0.6017343684
```

结论：

在 clean-source 条件下，speaker identity channel 足够稳定。

注意：

这并不等于解决 `similar_content` 的内容错配检测。它只说明 identity verifier 可以作为辅助通道。

## 5. 下一步判断

可以进入 Stage 3：

```text
基于模型输出的 local mismatch pilot
```

需要计算：

```text
utterance SI-SDR
local SI-SDR target/interferer gap
local ECAPA speaker gap
mismatch segment statistics
```

---

## 6. 解读摘要

这一步的结论是：

> ECAPA 在 clean target / clean interferer 的 1s 和 2s chunk 上可以稳定区分说话人。

它不是最终模型实验，而是为后续局部 mismatch 指标提供基础。

重要含义：

1. 1s chunk 已经可用。
2. 2s chunk 更稳，可以作为主报告尺度或稳健性检查。
3. speaker prototype 的判别余量较大，适合作为 local identity anchor。
4. 在 similar speaker hard condition 下仍然接近完美 AUC，说明 verifier 本身不是明显瓶颈。

限制：

1. 这还没有用模型输出。
2. 这还没有证明 output chunk 会发生 mismatch。
3. similar content 的内容错配仍需要 content verifier 或 SSL/phonetic 表征。

