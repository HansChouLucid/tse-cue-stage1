# 下一步：基于模型输出的 local mismatch pilot

## 1. 目标

在 clean-source sanity check 之后，下一步要验证：

1. 模型输出上的 local ECAPA gap 是否稳定
2. local ECAPA gap 是否和 local SI-SDR gap 有关系
3. mismatch 是否确实是局部且有限的
4. 是否存在 utterance SI-SDR 掩盖局部错误的样本

---

## 2. 需要的输入

### 2.1 数据 manifest

建议先用 test split：

```text
/data/tse_cue_project/manifests/librimix_similar_speaker_960h/clean/test/samples.jsonl
/data/tse_cue_project/manifests/librimix_similar_content_960h/clean/test/samples.jsonl
```

### 2.2 模型输出

需要已有或新跑的推理输出：

```text
/data/tse_cue_project/experiments/{model_name}/infer/{condition}/
```

每条输出应能按 `key` 对齐。

### 2.3 clean source

manifest 中已经包含：

```text
src[target_spk]
src[interferer_spk]
```

可用于 oracle local mismatch diagnosis。

---

## 3. 第一版指标

### 3.1 utterance-level

```text
utterance_si_sdr_target
utterance_si_sdr_interferer
utterance_si_sdr_gap
```

### 3.2 chunk-level

```text
local_si_sdr_target
local_si_sdr_interferer
local_si_sdr_gap

speaker_sim_target
speaker_sim_interferer
speaker_gap

speaker_mismatch = speaker_gap < 0
```

### 3.3 segment-level

```text
mismatch_rate
mismatch_segment_count
max_mismatch_duration
mean_mismatch_duration
mismatch_coverage
first_mismatch_position
```

---

## 4. 关键分析

### 4.1 局部性检验

看：

```text
mismatch_coverage
max_mismatch_duration
mismatch_segment_count
```

如果 mismatch 只占少量局部片段，则支持“局部且有限”的假设。

### 4.2 与传统指标对比

看：

```text
utterance SI-SDR vs mismatch_rate
utterance SI-SDR vs max_mismatch_duration
local SI-SDR gap vs speaker_gap
```

重点找：

```text
SI-SDR 不差但 mismatch_rate 高的样本
```

这类样本能证明局部指标的价值。

### 4.3 hard condition 对比

比较：

```text
normal
similar speaker
similar content
```

预期：

- similar speaker: speaker mismatch rate 更高
- similar content: 仅 speaker gap 可能不够，需要后续 content verifier

---

## 5. 先跑规模

第一版建议：

```text
100 samples
1.0s chunk
2.0s chunk
```

确认无误后扩展到：

```text
test full split
```

---

## 6. 预期输出

```text
per_chunk_metrics.jsonl
per_utterance_summary.csv
aggregate_report.json
top_mismatch_cases.csv
```

其中 `top_mismatch_cases.csv` 用于人工听和画图。

---

## 7. 已完成：smoke 模型链路验证

远端脚本：

```text
/data/tse_cue_project/code/wesep-real-tse/tools/run_model_output_mismatch_pilot.py
```

远端输出：

```text
/data/tse_cue_project/experiments/model_output_mismatch_pilot/smoke_global_spkemb_similar_speaker_test_50
```

使用 checkpoint：

```text
/data/tse_cue_project/experiments/smoke_tse_bsrnn_spk_librimix_similar/models/checkpoint_1.pt
```

注意：

该 checkpoint 只是 smoke 训练产物，不是正式 baseline。该实验只用于验证“模型输出 -> 局部指标”的评测链路是否跑通。

### 7.1 设置

```text
condition = similar_speaker test
num manifest samples = 50
num target directions = 100
chunk = 1.0s, 2.0s
verifier = ECAPA speaker prototype
```

### 7.2 结果摘要

```text
1.0s:
  speaker_gap mean = 0.0005
  speaker_gap median = -0.0023
  speaker_gap p10 = -0.2366
  speaker_gap p90 = 0.2340
  local_sisdr_gap mean = 0.0336
  mismatch_rate mean = 0.4985

2.0s:
  speaker_gap mean = 0.0017
  speaker_gap median = -0.0037
  speaker_gap p10 = -0.2466
  speaker_gap p90 = 0.2434
  local_sisdr_gap mean = 0.1363
  mismatch_rate mean = 0.5083
```

### 7.3 解读

这个 smoke 模型的输出在局部 identity gap 上接近随机：

```text
mismatch_rate ≈ 50%
speaker_gap mean ≈ 0
local_sisdr_gap mean ≈ 0
```

这说明它并没有形成可靠的 target extraction 能力。

因此不能把这个结果解释为“局部有限 mismatch”。它更像是：

> smoke checkpoint 训练不足，模型输出在 target/interferer 之间没有稳定倾向。

### 7.4 这一步的价值

尽管 smoke 模型本身不可用于正式结论，但这一步确认了：

1. 模型输出可以直接接入局部指标脚本。
2. 脚本能同时输出 local SI-SDR gap 与 ECAPA speaker gap。
3. `mismatch_rate / coverage / segment` 统计链路可运行。
4. 后续换成正式训练模型后，可以复用同一套评测。

### 7.5 下一步

需要训练或获得一个更可靠的 baseline / fine-grained 模型输出。

优先顺序：

1. 先训练一个 `global spkemb baseline`，至少在 train-clean-100 上跑到可用。
2. 在 `similar_speaker test` 上做 Stage 3 full pilot。
3. 再训练/启用 `fine-grained cue` 模型。
4. 比较两者 mismatch segment 是否呈现局部有限结构。
