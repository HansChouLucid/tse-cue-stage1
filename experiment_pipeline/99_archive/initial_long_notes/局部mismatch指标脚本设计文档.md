# 局部 mismatch 指标脚本设计文档

## 1. 脚本目标

设计一个可复用的离线评测脚本，用于在 TSE 模型输出之后，按 chunk 计算局部 mismatch 指标。

该脚本不负责训练模型，只负责分析：

1. 输出局部 chunk 是否更像 target 还是 interferer
2. hard condition 是否增加局部错配
3. similar speaker 与 similar content 是否呈现不同错配模式
4. attention / interaction map 是否能解释错配发生位置

---

## 2. 建议脚本名称

```text
tools/evaluate_local_mismatch.py
```

后续可以拆成多个模块：

```text
tools/evaluate_local_mismatch.py
tools/mismatch_metrics/speaker.py
tools/mismatch_metrics/content.py
tools/mismatch_metrics/attention.py
tools/mismatch_metrics/report.py
```

第一版建议先做单文件，跑通后再拆。

---

## 3. 输入

### 3.1 必需输入

```text
--samples-jsonl
```

来自数据集 manifest，例如：

```text
/data/tse_cue_project/manifests/librimix_similar_speaker_960h/clean/test/samples.jsonl
/data/tse_cue_project/manifests/librimix_similar_content_960h/clean/test/samples.jsonl
```

每条样本需要包含：

```json
{
  "key": "...",
  "condition": "similar_speaker",
  "spk": ["target_spk", "interferer_spk"],
  "mix": {"default": [".../mix_clean/key.wav"]},
  "src": {
    "target_spk": [".../s1/key.wav"],
    "interferer_spk": [".../s2/key.wav"]
  },
  "source_meta": {
    "target_source": "...",
    "interferer_source": "..."
  }
}
```

```text
--estimate-dir
```

模型输出目录。脚本默认按 `key.wav` 查找估计语音：

```text
estimate_dir/key.wav
```

如果推理脚本输出结构不同，可以增加：

```text
--estimate-pattern "{key}.wav"
```

### 3.2 verifier 输入

```text
--speaker-verifier-config
```

用于局部身份检测。第一版建议复用 WeSpeaker ECAPA。

```text
--content-verifier-type
```

用于局部内容检测。第一版可选：

```text
ssl
mfcc
asr_ctc
none
```

建议第一版先实现：

```text
mfcc
ssl
```

其中 MFCC 作为低成本 baseline，SSL 作为更强表征。

### 3.3 可选输入

```text
--attention-dir
```

如果模型保存了 attention / TF interaction map，则读取并计算解释指标。

```text
--condition
```

只评估某一类样本：

```text
similar_speaker
similar_content
normal
both
```

```text
--max-samples
```

用于 smoke test。

---

## 4. 输出

### 4.1 per-chunk 指标

```text
out_dir/per_chunk_metrics.jsonl
```

每行对应一个 chunk：

```json
{
  "key": "...",
  "condition": "similar_speaker",
  "chunk_idx": 3,
  "start_sec": 1.5,
  "end_sec": 2.5,
  "target_active": true,
  "interferer_active": true,
  "speaker_sim_target": 0.61,
  "speaker_sim_interferer": 0.48,
  "speaker_gap": 0.13,
  "speaker_mismatch": false,
  "content_sim_target": 0.42,
  "content_sim_interferer": 0.53,
  "content_gap": -0.11,
  "content_mismatch": true,
  "attention_shift_score": 0.27,
  "confidence_drop": 0.18
}
```

### 4.2 per-utterance 汇总

```text
out_dir/per_utterance_summary.csv
```

字段建议：

```text
key
condition
num_chunks
speaker_mismatch_rate
content_mismatch_rate
mean_speaker_gap
min_speaker_gap
mean_content_gap
min_content_gap
max_mismatch_duration_sec
first_mismatch_sec
mean_attention_shift
```

### 4.3 aggregate 报告

```text
out_dir/aggregate_report.json
```

按 condition 聚合：

```json
{
  "similar_speaker": {
    "num_utts": 6000,
    "speaker_mismatch_rate_mean": 0.18,
    "content_mismatch_rate_mean": 0.07,
    "mean_speaker_gap": 0.12,
    "max_mismatch_duration_sec_mean": 0.9
  }
}
```

### 4.4 可视化输出

第一版可选生成：

```text
out_dir/plots/{key}_timeline.png
```

包含：

1. speaker gap 曲线
2. content gap 曲线
3. mismatch mask
4. attention shift 曲线
5. target/interferer activity

---

## 5. Chunk 切分协议

建议参数：

```text
--chunk-sec 1.0
--hop-sec 0.5
--sample-rate 16000
```

同时支持多尺度：

```text
--chunk-sec 0.5 1.0 2.0
```

第一版实现可以只支持单个 chunk size，后续扩展多尺度。

### 5.1 chunk 有效性

chunk 太短或能量太低时，speaker verifier 不稳定。

建议加入：

```text
--min-rms-db -45
--min-active-ratio 0.3
```

低能量 chunk 标记为：

```text
"valid_for_speaker": false
```

而不是直接判 mismatch。

---

## 6. Speaker mismatch 指标

### 6.1 局部身份相似度

对每个 chunk：

```text
e_y(t) = speaker_encoder(output_chunk_t)
e_tar(t) = speaker_encoder(target_chunk_t)
e_int(t) = speaker_encoder(interferer_chunk_t)
```

计算：

```text
speaker_sim_target(t) = cos(e_y(t), e_tar(t))
speaker_sim_interferer(t) = cos(e_y(t), e_int(t))
speaker_gap(t) = speaker_sim_target(t) - speaker_sim_interferer(t)
```

判定：

```text
speaker_mismatch(t) = speaker_gap(t) < margin_id
```

默认：

```text
margin_id = 0.0
```

更保守时：

```text
margin_id = -0.05
```

### 6.2 prototype 版本

chunk 太短时，可以用 target/interferer 的 speaker prototype 替代 clean chunk embedding：

```text
speaker_gap_proto(t) = cos(e_y(t), proto_target) - cos(e_y(t), proto_interferer)
```

prototype 可来自：

1. enrollment cue
2. 同 speaker 多 utterance
3. 数据构造时的 speaker inventory

第一版建议同时输出：

```text
speaker_gap_chunk
speaker_gap_proto
```

后续比较哪一个更稳定。

---

## 7. Content mismatch 指标

### 7.1 局部内容表示

第一版建议先实现两种：

#### MFCC baseline

```text
c_y(t) = mean_pool(MFCC(output_chunk_t))
c_tar(t) = mean_pool(MFCC(target_chunk_t))
c_int(t) = mean_pool(MFCC(interferer_chunk_t))
```

优点是轻量，缺点是容易受 speaker/timbre 影响。

#### SSL representation

用 WavLM / HuBERT / wav2vec2 的 frame-level hidden states：

```text
c_y(t) = mean_pool(SSL(output_chunk_t))
```

优点是更接近 phonetic/content，缺点是依赖模型和计算开销。

### 7.2 内容 gap

```text
content_sim_target(t) = cos(c_y(t), c_tar(t))
content_sim_interferer(t) = cos(c_y(t), c_int(t))
content_gap(t) = content_sim_target(t) - content_sim_interferer(t)
```

判定：

```text
content_mismatch(t) = content_gap(t) < margin_content
```

默认：

```text
margin_content = 0.0
```

### 7.3 DTW / alignment 扩展

如果 chunk 内语速差异明显，可以用 DTW 对齐 frame-level SSL feature：

```text
dtw_dist(output_chunk, target_chunk)
dtw_dist(output_chunk, interferer_chunk)
content_gap_dtw = dist_to_interferer - dist_to_target
```

第一版不强制实现，作为第二版扩展。

---

## 8. Attention / interaction map 指标

attention 不建议作为最终判定，但适合解释 mismatch。

### 8.1 Attention Shift Score

如果模型能输出 attention map：

```text
A(t_mix, t_ref)
```

可以计算：

```text
attention_entropy(t)
attention_peakiness(t)
attention_temporal_shift(t)
```

如果有 target/interferer active mask，可进一步计算 attention 是否集中在干扰活跃区域。

### 8.2 Interaction Map Entropy

对 T-F similarity / interaction map：

```text
M(f, t)
```

计算：

```text
entropy(M_t)
max_confidence(M_t)
temporal_instability(M_t)
```

hard condition 下，如果 interaction map 更尖锐地响应 interferer-like 区域，可能说明 fine-grained cue 被误导。

---

## 9. 连续性指标

单个 chunk 的误判可能是 verifier 噪声，所以需要连续性过滤。

### 9.1 mismatch segment

把连续的 mismatch chunk 合并成 segment：

```text
[chunk_i, chunk_j]
```

输出：

```text
num_mismatch_segments
max_mismatch_duration_sec
mean_mismatch_duration_sec
first_mismatch_sec
```

### 9.2 平滑规则

建议：

```text
min_consecutive_chunks = 2
```

只有连续两个及以上 chunk 都 mismatch，才记为 stable mismatch。

---

## 10. Target/interferer activity

局部检测必须考虑源是否 active。

建议通过 clean source RMS 得到 activity：

```text
target_active(t) = rms(target_t) > threshold
interferer_active(t) = rms(interferer_t) > threshold
```

只在以下情况重点分析：

```text
target_active = true
interferer_active = true
```

如果 target 不活跃而 output 更像 interferer，可能是 target-absent false extraction，不应和普通 local mismatch 混在一起。

---

## 11. 推荐命令

### 11.1 similar speaker test

```bash
python tools/evaluate_local_mismatch.py \
  --samples-jsonl /data/tse_cue_project/manifests/librimix_similar_speaker_960h/clean/test/samples.jsonl \
  --estimate-dir /data/tse_cue_project/experiments/MODEL/infer/similar_speaker_test \
  --out-dir /data/tse_cue_project/experiments/MODEL/mismatch/similar_speaker_test \
  --chunk-sec 1.0 \
  --hop-sec 0.5 \
  --speaker-verifier ecapa \
  --content-verifier none
```

### 11.2 similar content test

```bash
python tools/evaluate_local_mismatch.py \
  --samples-jsonl /data/tse_cue_project/manifests/librimix_similar_content_960h/clean/test/samples.jsonl \
  --estimate-dir /data/tse_cue_project/experiments/MODEL/infer/similar_content_test \
  --out-dir /data/tse_cue_project/experiments/MODEL/mismatch/similar_content_test \
  --chunk-sec 1.0 \
  --hop-sec 0.5 \
  --speaker-verifier ecapa \
  --content-verifier ssl
```

---

## 12. 实现顺序

### Version 0: 能跑

实现：

1. 读取 `samples.jsonl`
2. 读取 output / target / interferer wav
3. chunk 切分
4. RMS activity
5. speaker gap
6. per-chunk jsonl
7. per-utterance csv

暂不做 content verifier 和 attention。

### Version 1: 支持 similar content

增加：

1. MFCC content gap
2. SSL content gap
3. content mismatch rate

### Version 2: 支持解释分析

增加：

1. attention map 读取
2. attention shift score
3. timeline 可视化

---

## 13. 自我思路检查

### 13.1 这种方式的优势

#### 优势 1：真正把问题局部化

它不是只看整段输出像不像目标，而是看每个 chunk 是否发生身份或内容偏移。这样更贴合 local mismatch 的研究问题。

#### 优势 2：可以区分两类 hard condition

`similar speaker` 用 identity gap 作为主指标。
`similar content` 用 content gap 作为主指标。

这样不会把两类不同错误都粗暴归为 speaker confusion。

#### 优势 3：兼容 oracle 分析和实际检测

有 clean target/interferer 时，可以计算 oracle gap。
没有 interferer 时，也可以退化成 target consistency / confidence drop。

#### 优势 4：能解释，不只是报分

attention、interaction map、temporal drift 可以帮助解释 mismatch 是在哪里、为什么发生。

#### 优势 5：便于和模型结构对齐

global-only、fine-grained-only、fusion、verifier 都可以用同一套指标评估。

### 13.2 可能出现的问题

#### 问题 1：短 chunk speaker embedding 不稳定

speaker verifier 原本常用于较长 utterance。0.5s 或 1s chunk 可能不稳定。

缓解：

- 多尺度评估：0.5s / 1s / 2s
- 加 activity mask
- 用连续性规则过滤孤立误判
- 同时输出 prototype gap 和 clean chunk gap

#### 问题 2：content verifier 可能混入 speaker 信息

MFCC 或 SSL 表征不一定是纯内容，也可能保留音色。

缓解：

- MFCC 只作为 baseline
- 优先尝试 SSL 中间层
- 后续可引入 ASR CTC / phone posterior
- 不把 content gap 单独解释为语义错误，而解释为局部声学/音素内容归属偏移

#### 问题 3：attention 不一定可解释

attention 高不等于模型真的依赖那里。

缓解：

- attention 只作为辅助解释信号
- 最终 mismatch 判定仍以 output 与 target/interferer 的相似度 gap 为主

#### 问题 4：oracle clean source 依赖强

第一阶段可以用 clean source 做诊断，但真实推理时没有 interferer。

缓解：

- 明确区分 oracle diagnosis 和 deployable detection
- 第一阶段先证明 failure mode
- 第二阶段再把 target-only verifier / confidence drop 作为可部署信号

#### 问题 5：阈值选择会影响结论

`margin = 0` 不一定最稳。

缓解：

- 报连续曲线，而不是只报单阈值结果
- 用 dev set 校准 margin
- 同时报告 AUC / PR-AUC

### 13.3 关键性假设

#### 假设 1：chunk-level 表征仍包含足够身份/内容信息

如果 chunk 太短或能量太低，身份和内容判断都会不稳定。因此必须有 activity mask 和多尺度评估。

#### 假设 2：clean target/interferer 可以作为局部归属参照

合成数据中有干净源，所以可以定义 oracle local mismatch。这是第一阶段 failure-mode analysis 的基础。

#### 假设 3：hard condition 会改变局部 gap 分布

如果 similar speaker / similar content 真的诱发 mismatch，那么它们的 gap 分布应当比 normal condition 更差。

#### 假设 4：speaker mismatch 和 content mismatch 是可分的

similar speaker 主要表现为身份偏移。
similar content 主要表现为内容/音素归属偏移。
实际中二者可能耦合，所以需要同时输出 identity gap 和 content gap。

#### 假设 5：attention / interaction map 与局部错误有统计相关性

attention 不是因果证据，但如果 hard condition 下 attention shift 与 mismatch segment 共现，就能作为解释性证据。

---

## 14. 审查重点

需要重点确认：

1. chunk 长度是否先用 1.0s / 0.5s hop
2. speaker verifier 是否先用 ECAPA
3. content verifier 第一版用 MFCC 还是 SSL
4. mismatch 判定是否先用 `gap < 0`
5. 是否接受 attention 只作为解释信号

