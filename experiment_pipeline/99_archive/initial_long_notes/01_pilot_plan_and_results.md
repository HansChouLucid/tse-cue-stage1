# 局部 mismatch 先导实验计划

## 0. 目的

在正式训练/大规模评测之前，先做一组小实验验证两个关键问题：

1. mismatch 是否确实只发生在局部、有限片段，而不是整段都错。
2. 当前选择的 verifier 是否能在 chunk 级别稳定地区分 target / interferer。

这一步的目标不是证明模型多强，而是证明后续局部指标体系是否可靠。

---

## 1. 核心判断

### 1.1 mismatch 应该是局部且稀疏的

你的判断是合理的。TSE 中的局部 mismatch 不应被默认理解为整段 speaker confusion。

更合理的现象应该是：

- 大部分 chunk 仍然接近 target
- 在 target/interferer 同时活跃、且局部声学或内容相似度高的区域，才出现短时偏移
- mismatch segment 的数量有限，持续时间有限
- hard condition 下 mismatch rate 上升，但不是全局崩溃

因此实验必须报告：

```text
chunk-level mismatch rate
max mismatch duration
mean mismatch duration
first mismatch position
mismatch segment count
```

而不是只报告一个 utterance-level score。

### 1.2 verifier 必须先做 chunk-level sanity check

如果 ECAPA / SV verifier 在 0.5s / 1s / 2s chunk 上本身区分不了 target 和 interferer，那么后面的 mismatch 指标就站不稳。

所以第一步应该先不使用模型输出，而是只用 clean source 做 verifier 判别性测试。

---

## 2. 实验 A：chunk-level verifier 判别性测试

### 2.1 输入

使用已经构造好的数据：

```text
similar_speaker test
similar_content test
normal/libri2mix test
```

每条样本有：

```text
target clean source
interferer clean source
enrollment / speaker prototype
```

### 2.2 做法

对 target 和 interferer 按窗口切 chunk：

```text
chunk = 0.5s, 1.0s, 2.0s
hop = chunk / 2
```

计算：

```text
sim(target_chunk, target_proto)
sim(target_chunk, interferer_proto)
sim(interferer_chunk, target_proto)
sim(interferer_chunk, interferer_proto)
```

然后看 verifier 能不能把 target chunk 和 interferer chunk 分开。

### 2.3 指标

```text
target_margin = sim(target_chunk, target_proto) - sim(target_chunk, interferer_proto)
interferer_margin = sim(interferer_chunk, interferer_proto) - sim(interferer_chunk, target_proto)
```

报告：

```text
mean margin
median margin
p10 margin
chunk-level accuracy
AUC
EER
invalid low-energy chunk ratio
```

### 2.4 判断标准

如果：

```text
1s chunk AUC > 0.8
2s chunk AUC > 0.85
margin 分布在 similar speaker 下仍然有可分性
```

那么 ECAPA verifier 可以作为第一版局部身份检测器。

如果：

```text
0.5s 很差，但 1s/2s 稳定
```

则后续主指标用 1s 或 2s，0.5s 只做补充。

如果：

```text
similar speaker 下 AUC 接近随机
```

说明 ECAPA 在 hard set 上区分力不足，需要换 verifier 或用 prototype/multi-scale。

---

## 3. 实验 B：局部指标与传统 SI-SDR 的关系

### 3.1 目的

验证局部 mismatch 指标是否提供了传统 SI-SDR 看不到的信息。

### 3.2 输入

使用现有模型输出：

```text
output
target clean
interferer clean
```

先从已有 fine-grained cue 模型或 baseline 输出开始，不需要新模型。

### 3.3 传统指标

整段：

```text
utterance SI-SDR
SDR
STOI/PESQ if available
```

局部：

```text
local SI-SDR(output_chunk, target_chunk)
local SI-SDR(output_chunk, interferer_chunk)
```

### 3.4 局部 mismatch 指标

```text
speaker_gap(t)
content_gap(t)
mismatch_rate
max_mismatch_duration
first_mismatch_position
```

### 3.5 关键对比

按 utterance 统计：

```text
SI-SDR vs mismatch_rate
SI-SDR vs max_mismatch_duration
SI-SDR vs min_speaker_gap
local SI-SDR target/interferer gap vs verifier gap
```

需要观察：

1. 是否存在 SI-SDR 不差但 mismatch_rate 高的样本
2. hard condition 下 mismatch_rate 是否比 normal 更敏感
3. local SI-SDR 是否能和 verifier gap 对齐
4. speaker mismatch 和 content mismatch 是否呈现不同模式

### 3.6 预期结果

理想情况下：

- normal condition: SI-SDR 高，mismatch_rate 低
- similar speaker: SI-SDR 可能下降，speaker_mismatch_rate 上升
- similar content: SI-SDR 可能未显著下降，但 content_mismatch_rate 上升
- 部分样本出现“整体还行，但局部错”的现象

这将支持你的核心判断：mismatch 是局部有限的，不是全局崩溃。

---

## 4. 实验 C：mismatch 是否局部且有限

### 4.1 做法

对每条输出生成二值时间序列：

```text
mismatch_mask[t] = 1 if gap(t) < threshold else 0
```

合并连续 mismatch chunk，得到 mismatch segments。

### 4.2 指标

```text
mismatch_segment_count
max_segment_duration
mean_segment_duration
total_mismatch_duration
mismatch_coverage = total_mismatch_duration / utterance_duration
```

### 4.3 判断标准

如果大多数样本满足：

```text
mismatch_coverage < 30%
segment_count 有限
max_segment_duration 不覆盖全句
```

则说明 mismatch 确实是局部现象。

如果大量样本：

```text
mismatch_coverage > 70%
```

那更像 global target confusion，不应再叫 local mismatch。

---

## 5. 实验 D：similar speaker 与 similar content 的差异

### 5.1 Similar speaker

重点看：

```text
speaker_gap
speaker_mismatch_rate
identity drift
```

预期：

```text
speaker_gap 分布整体左移
speaker_mismatch_rate 上升
content_gap 不一定显著恶化
```

### 5.2 Similar content

重点看：

```text
content_gap
content_mismatch_rate
phonetic/SSL similarity shift
```

预期：

```text
content_gap 更敏感
speaker_gap 可能不明显恶化
attention 或 interaction 更容易被相似内容吸引
```

### 5.3 两者叠加

预期：

```text
speaker_gap 和 content_gap 同时恶化
mismatch segment 更长
first mismatch 更早
```

---

## 6. 第一版必须先实现的最小指标

为了尽快验证思路，第一版不要太复杂。

先实现：

```text
1. chunk 切分
2. RMS activity mask
3. speaker verifier gap
4. local SI-SDR target/interferer gap
5. mismatch segment 统计
6. utterance SI-SDR 与 mismatch 指标相关性
```

暂缓：

```text
content SSL verifier
attention map
timeline 可视化
```

原因是：先确认 chunk-level speaker verifier 和 local SI-SDR 是否能形成稳定信号。

---

## 7. 自我检查

### 7.1 优势

1. 先验证 verifier 本身，不会把 verifier 不稳定误认为模型 mismatch。
2. 用 local SI-SDR 作为传统指标对照，能说明新指标不是凭空造的。
3. segment 统计能直接验证“局部且有限”这个假设。
4. similar speaker 和 similar content 分开看，避免把不同错误混在一起。

### 7.2 风险

1. ECAPA 在短 chunk 上可能不稳。
2. similar speaker 本来就是高难身份区分，verifier margin 可能很小。
3. local SI-SDR 对短窗口也会波动，需要 activity mask。
4. content mismatch 第一版如果不用 SSL/ASR，可能还不能充分验证 similar content。

### 7.3 关键假设

1. 1s 或 2s chunk 中仍然有足够身份信息。
2. clean target/interferer 可以作为局部 oracle 参照。
3. local SI-SDR target/interferer gap 可以作为传统信号对照。
4. hard condition 会改变局部 gap/segment 分布，而不仅仅降低整段 SI-SDR。

---

## 8. 建议审批点

建议先批准以下最小实验：

1. 在 `similar_speaker test` 和 `similar_content test` 上做 clean-source verifier sanity check。
2. chunk size 用 `1.0s` 和 `2.0s`。
3. verifier 先用 ECAPA。
4. 同时计算 local SI-SDR gap。
5. 输出 per-chunk 和 per-utterance 两张表。

如果这一步结果稳定，再进入 content SSL verifier 和 attention 分析。

---

## 9. 已完成的初步 sanity 结果

远端实例：

```text
js4.blockelite.cn:13216
GPU: A100-SXM4-80GB
```

脚本：

```text
/data/tse_cue_project/code/wesep-real-tse/tools/run_local_mismatch_pilot.py
```

输出目录：

```text
/data/tse_cue_project/experiments/local_mismatch_pilot
```

### 9.1 similar speaker test, 1000 samples

设置：

```text
chunk = 1.0s, 2.0s
hop = chunk / 2
verifier = ECAPA
prototype utterances = 3
```

结果：

```text
1.0s chunk:
  AUC = 0.9998
  accuracy(gap > 0) = 0.9944
  target margin mean = 0.3307
  target margin p10 = 0.1794
  interferer margin mean = 0.3248
  interferer margin p10 = 0.1775

2.0s chunk:
  AUC = 0.99999
  accuracy(gap > 0) = 0.9977
  target margin mean = 0.3981
  target margin p10 = 0.2642
  interferer margin mean = 0.3948
  interferer margin p10 = 0.2640
```

### 9.2 similar content test, 100 samples

结果：

```text
1.0s chunk:
  AUC = 0.99999
  accuracy(gap > 0) = 0.9991
  target margin mean = 0.5000

2.0s chunk:
  AUC = 1.0000
  accuracy(gap > 0) = 1.0000
  target margin mean = 0.6017
```

### 9.3 当前结论

ECAPA verifier 在 clean-source chunk 级别具有足够强的 target/interferer 判别力。

这说明：

1. `1.0s` 和 `2.0s` chunk 可以作为第一版局部身份检测尺度。
2. 使用 speaker prototype 比直接比较短 chunk 更稳定。
3. 后续如果模型输出发生局部身份偏移，ECAPA gap 有希望检测出来。

但要注意：

1. 这只是 clean-source sanity check，还不是模型输出 mismatch 检测。
2. clean-source 判别强，不代表 noisy/separated output 上同样稳定。
3. similar content 的核心仍然需要 content verifier；ECAPA 只能说明身份判别通道可用。

### 9.4 下一步

下一步应该基于现有 fine-grained/baseline 模型输出，计算：

```text
utterance SI-SDR
local SI-SDR target/interferer gap
local ECAPA speaker gap
mismatch segment statistics
```

重点验证：

```text
局部指标是否能发现 utterance SI-SDR 平均掉的局部错误。
```

---

## 10. 指标详细解释

这一阶段是 verifier sanity check，不是模型效果评测。它回答的是：

> 如果我们把 clean target / clean interferer 切成局部 chunk，ECAPA verifier 是否仍然能判断这个 chunk 属于谁。

### 10.1 AUC

`AUC` 衡量 verifier 把 target chunk 和 interferer chunk 分开的能力。

```text
AUC = 1.0  表示完美区分
AUC = 0.5  表示随机猜测
```

本实验中：

```text
similar speaker, 1.0s: AUC = 0.9998
similar speaker, 2.0s: AUC = 0.99999
```

说明即使在声纹相似的 hard set 中，ECAPA 在 clean-source 局部 chunk 上仍具有非常强的身份判别能力。

### 10.2 accuracy(gap > 0)

这里的 `gap` 是：

```text
gap = sim(chunk, correct_speaker_prototype) - sim(chunk, wrong_speaker_prototype)
```

如果 `gap > 0`，说明 chunk 更接近正确说话人的 prototype。

本实验中：

```text
similar speaker, 1.0s accuracy = 0.9944
similar speaker, 2.0s accuracy = 0.9977
```

说明绝大多数局部 chunk 都被正确判别。

### 10.3 target margin mean / p10

`target margin mean` 是 target chunk 的平均判别余量。

```text
target_margin = sim(target_chunk, target_proto) - sim(target_chunk, interferer_proto)
```

`p10` 是第 10 百分位，比 mean 更保守。

如果 p10 仍然大于 0，说明不是少数容易样本拉高了平均值，而是大多数 chunk 都稳定。

similar speaker 1000 条样本结果：

```text
1.0s target margin mean = 0.3307
1.0s target margin p10  = 0.1794

2.0s target margin mean = 0.3981
2.0s target margin p10  = 0.2642
```

解读：

- 2.0s 比 1.0s 更稳，因为包含更多 speaker 信息。
- 1.0s 已经足够稳定，可以作为后续局部 mismatch 主尺度之一。
- p10 明显大于 0，说明 verifier 的 chunk-level 判别不是偶然。

### 10.4 interferer margin

`interferer margin` 是同样的判别式，只是换成 interferer chunk：

```text
interferer_margin = sim(interferer_chunk, interferer_proto) - sim(interferer_chunk, target_proto)
```

它用于验证 verifier 是否双向稳定，而不是只会识别 target。

similar speaker 1000 条样本结果：

```text
1.0s interferer margin mean = 0.3248
1.0s interferer margin p10  = 0.1775

2.0s interferer margin mean = 0.3948
2.0s interferer margin p10  = 0.2640
```

这说明 target 和 interferer 两侧的判别余量都稳定。

### 10.5 clean local SI-SDR gap

当前阶段的 local SI-SDR gap 只是流程检查。

因为这里拿的是 clean source 自己与自己比较：

```text
SI-SDR(target_chunk, target_chunk) - SI-SDR(target_chunk, interferer_chunk)
```

所以数值会非常大，不代表模型性能。

它的作用只是确认：

- local chunk 切分正常
- local SI-SDR 计算流程正常
- 后续可以用同一套代码处理模型输出

---

## 11. 这一步能说明什么

### 11.1 可以说明

1. `1.0s / 2.0s` chunk 是可行的局部身份分析尺度。
2. ECAPA verifier 在 clean-source 局部 chunk 上判别力足够强。
3. 后续如果模型输出 chunk 的 speaker gap 变差，不能简单归因于 verifier 本身完全不可用。
4. 使用 speaker prototype 比直接短 chunk 对短 chunk 更稳定。

### 11.2 还不能说明

1. 还不能证明模型真的发生了局部 mismatch。
2. 还不能证明 fine-grained cue 被 similar speaker / content 误导。
3. 还不能证明 local metric 比 utterance-level SI-SDR 更有解释力。
4. 还不能解决 similar content 的内容错配检测，因为当前主要验证的是 speaker identity channel。

---

## 12. 为什么下一步必须转向模型输出

clean-source sanity check 只是 verifier 的能力测试。

真正的研究问题是：

> 当模型从 mixture 中抽取 target speech 时，输出的某些局部片段是否会更接近 interferer。

因此下一阶段必须把输入换成模型输出：

```text
output_chunk
target_clean_chunk
interferer_clean_chunk
```

然后同时计算：

```text
utterance SI-SDR
local SI-SDR target/interferer gap
local ECAPA speaker gap
mismatch segment statistics
```

只有这样才能验证：

1. mismatch 是否局部且有限
2. utterance SI-SDR 是否会掩盖局部错误
3. local ECAPA gap 是否和 local SI-SDR gap 对齐
4. hard condition 是否让 mismatch segment 变多、变长或更早出现
