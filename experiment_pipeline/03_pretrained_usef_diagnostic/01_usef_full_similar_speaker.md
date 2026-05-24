# USEF-TFGridNet Full Similar-Speaker Diagnostic

## 目的

这个实验用于检查一个公开预训练的 fine-grained cue 模型在 hard similar-speaker 条件下是否存在局部 target/interferer mismatch。

当前不把它作为公平模型对比实验，而是作为诊断实验：

```text
如果一个整体表现很强的 fine-grained cue 模型仍然出现局部身份偏移，
那么局部 mismatch 就是值得单独研究的问题。
```

## 实验设置

模型：

```text
model: USEF-TFGridNet
source: public USEF-TSE pretrained model
checkpoint: /data/tse_cue_project/models/pretrained/USEF-TSE/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar
sample rate: 8000 Hz
```

数据：

```text
condition: similar_speaker
split: test
utterances: 6000
metadata: /data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_speaker_test_full/metadata.jsonl
```

远端输出目录：

```text
/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_speaker_test_full/usef_tfgridnet_wsj0_2mix
```

主要结果文件：

```text
inference_summary.csv
local_mismatch_ecapa/aggregate_summary.json
local_mismatch_ecapa/per_utterance_summary.csv
local_mismatch_ecapa/per_chunk_metrics.jsonl
failure_taxonomy_v2/taxonomy_summary_1s.json
failure_taxonomy_v2/taxonomy_summary_2s.json
failure_taxonomy_v2/utterance_taxonomy_1s.csv
failure_taxonomy_v2/utterance_taxonomy_2s.csv
```

## 整句 SI-SNR 结果

| Metric | Value |
|---|---:|
| test utterances | 6000 |
| mixture SI-SNR mean | -0.066 dB |
| estimated SI-SNR mean | 15.796 dB |
| SI-SNRi mean | 15.862 dB |
| SI-SNRi median | 18.491 dB |
| SI-SNRi p10 | 10.992 dB |
| SI-SNRi < 0 | 323 / 6000 |
| SI-SNRi < 5 | 404 / 6000 |
| SI-SNRi > 15 | 4943 / 6000 |

解释：

USEF-TFGridNet 在 full similar-speaker test 上整体表现很强。大多数样本有明显的 SI-SNR 提升，但仍有 323 条样本出现负提升。这些负提升样本是 waveform-level failure 的主要候选。

## 局部指标定义

对模型输出按 1s 和 2s chunk 进行检测。

局部说话人 gap：

```text
speaker_gap(t)
= cos(ECAPA(output_chunk_t), target_speaker_proto)
- cos(ECAPA(output_chunk_t), interferer_speaker_proto)
```

当 `speaker_gap(t) < 0` 时，该 chunk 在 ECAPA verifier 看来更像 interferer。

局部波形 gap：

```text
local_sisdr_gap(t)
= SI-SDR(output_chunk_t, target_chunk_t)
- SI-SDR(output_chunk_t, interferer_chunk_t)
```

当 `local_sisdr_gap(t) > 0` 时，该 chunk 从 waveform 角度更接近 target。

## Full-Test 局部结果

| Chunk | chunk rows | speaker_gap mean | speaker_gap p10 | local_sisdr_gap mean | local_sisdr_gap p10 | mismatch_rate mean | mismatch_rate median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1s | 53114 | 0.044 | -0.0566 | 46.40 dB | 33.85 dB | 0.291 | 0.231 |
| 2s | 22324 | 0.0619 | -0.0409 | 47.16 dB | 35.48 dB | 0.222 | 0.000 |

关键观察：

1. `local_sisdr_gap` 整体很高，说明输出波形大多仍然更接近 target。
2. `speaker_gap` 的 p10 为负，说明有一部分局部 chunk 在身份表征上更接近 interferer。
3. 1s 的 mismatch rate 高于 2s，说明很多 mismatch 是短时局部现象，容易被更长窗口平滑掉。

## 修正后的 Failure Taxonomy

v2 taxonomy 避免把所有高 mismatch-rate 样本都叫做 global failure。当前分类如下：

| Label | 含义 |
|---|---|
| `waveform_global_failure` | 整句 SI-SNRi < 0，模型从 waveform 指标上整体失败 |
| `global_identity_disagreement` | SI-SNRi 较高，但大多数 chunk 的 speaker verifier 更偏向 interferer |
| `true_local_swap` | 只有部分 chunk 同时满足 `speaker_gap < 0` 和 `local_sisdr_gap < 0` |
| `local_identity_ambiguity` | 部分 chunk 满足 `speaker_gap < 0`，但 `local_sisdr_gap` 仍为正 |
| `mixed_or_mild_mismatch` | 有轻微或混合型异常，但不满足以上强条件 |
| `strong_success` | SI-SNRi 高且 speaker mismatch rate 低 |

### 1s taxonomy

| Label | Count |
|---|---:|
| `strong_success` | 1333 |
| `local_identity_ambiguity` | 2935 |
| `mixed_or_mild_mismatch` | 1108 |
| `global_identity_disagreement` | 265 |
| `true_local_swap` | 36 |
| `waveform_global_failure` | 323 |

### 2s taxonomy

| Label | Count |
|---|---:|
| `strong_success` | 3061 |
| `local_identity_ambiguity` | 1560 |
| `mixed_or_mild_mismatch` | 624 |
| `global_identity_disagreement` | 428 |
| `true_local_swap` | 4 |
| `waveform_global_failure` | 323 |

## 研究解读

这个结果支持三个判断。

第一，USEF-TFGridNet 的整体提取能力很强，所以局部 mismatch 不是因为模型完全不可用。它在大多数样本上可以稳定提升 SI-SNR。

第二，局部身份 ambiguity 很明显。1s 下有 2935 条样本被归为 `local_identity_ambiguity`，说明 output 在 waveform 层面接近 target，但局部 speaker identity evidence 会短暂偏向 interferer。这正是 fine-grained cue mismatch 研究最值得挖的现象。

第三，真正的 waveform-level local swap 较少。1s 下 `true_local_swap` 为 36 条，2s 下为 4 条，说明多数问题不是“模型真的把局部波形提成 interferer”，而是更细的身份判别不稳定或 cue attribution ambiguity。

## 需要谨慎的地方

`global_identity_disagreement` 不能直接解释为模型失败。它表示 SI-SNRi 较高，但 ECAPA verifier 在大多数 chunk 上更偏向 interferer。可能原因包括：

1. target 和 interferer 声纹相似，verifier 本身边界不稳定；
2. USEF 输出保留了足够 target waveform，但 speaker embedding 被局部音色或内容扰动影响；
3. enrollment cue、target prototype、chunk 长度共同影响了 verifier 分数。

因此后续需要用 case study 和多 verifier 验证，避免把 verifier disagreement 直接写成模型 swap。

## 下一步

1. 从 `failure_taxonomy_v2` 中挑选每类代表 case。
2. 为代表 case 生成 timeline 图：
   ```text
   speaker_gap(t)
   local_sisdr_gap(t)
   target/interferer activity
   ```
3. 对 `global_identity_disagreement` 做人工听感和第二 verifier 复核。
4. 将同一套流程迁移到 `similar_content`，加入 content-level verifier。
