# Stage 3: 模型输出上的局部 Mismatch Pilot

## 目的

Stage 2 已证明 clean-source chunk 上 ECAPA verifier 足够稳定。

Stage 3 要回答：

```text
模型输出中是否存在局部、有限的 target/interferer 偏移？
```

---

## 子实验 3.1: Smoke 模型链路验证

### 目的

验证脚本链路能否从模型输出直接生成局部指标。

### 设置

| 项目 | 设置 |
|---|---|
| 模型 | `smoke_tse_bsrnn_spk_librimix_similar/checkpoint_1.pt` |
| 数据 | `similar_speaker test` |
| 样本 | 50 manifest samples，100 target directions |
| chunk | 1.0s / 2.0s |
| 指标 | local SI-SDR gap, ECAPA speaker gap, mismatch rate |

### 结果

| chunk | speaker gap mean | speaker gap median | local SI-SDR gap mean | mismatch rate mean |
|---|---:|---:|---:|---:|
| 1.0s | 0.0005 | -0.0023 | 0.0336 | 0.4985 |
| 2.0s | 0.0017 | -0.0037 | 0.1363 | 0.5083 |

### 解读

smoke checkpoint 几乎没有可靠 target extraction 能力：

```text
speaker_gap ≈ 0
mismatch_rate ≈ 50%
```

这不是“局部有限 mismatch”，而更像未充分训练模型的随机/不稳定输出。

### 价值

这一步确认了评测链路可运行：

- 可以直接加载模型输出
- 可以计算 local SI-SDR gap
- 可以计算 ECAPA speaker gap
- 可以统计 mismatch rate / segment

---

## 子实验 3.2: 正在训练可用 Global SpkEmb Baseline

### 目的

获得一个真正具备 target extraction 能力的模型输出，再做局部 mismatch 诊断。

### 设置

| 项目 | 设置 |
|---|---|
| 模型 | global spkemb baseline |
| 训练数据 | `similar_speaker train-100` |
| 验证数据 | `similar_speaker dev` |
| epoch | 5 |
| batch size | 2 |
| chunk_len | 2s |

远端配置：

```text
/data/tse_cue_project/experiments/configs/baseline_global_spkemb_similar_speaker_train100_5ep.yaml
```

远端实验目录：

```text
/data/tse_cue_project/experiments/baseline_global_spkemb_similar_speaker_train100_5ep
```

tmux：

```bash
tmux attach -t baseline_global_train100
```

### 当前进度

截至 2026-05-21 11:13：

```text
epoch 1
iter 2200 / 13900
train loss 已从 0.50191 降到 -0.19366
```

### 预计完成时间

当前速度约：

```text
100 iter / 25-30 秒
```

每个 epoch：

```text
13900 iter ≈ 58-70 分钟
```

5 epoch 加验证预计：

```text
约 5.5 - 6.5 小时
```

---

## 子实验 3.3: 训练完成后的局部指标评测

训练完成后执行：

```text
similar_speaker test
similar_content test
```

核心指标：

```text
utterance SI-SDR
local SI-SDR target/interferer gap
local ECAPA speaker gap
mismatch rate
mismatch coverage
max mismatch duration
first mismatch position
```

判断标准：

- 如果整体 SI-SDR 可用，但 mismatch coverage 较低且集中在少数 segment，说明进入真正的 local mismatch 分析阶段。
- 如果 mismatch rate 仍接近 50%，说明模型仍不可用，需要继续训练或调整 baseline。

