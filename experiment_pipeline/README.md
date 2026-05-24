# Audio-Only TSE Cue 实验流水线

## 核心研究问题

本项目关注：在 hard condition 下，fine-grained cue TSE 模型是否会产生局部、有限、可诊断的 target/interferer mismatch。

当前证据链按阶段推进：

```text
01 hard-condition 数据构造
02 chunk-level verifier sanity
03 USEF-only 模型输出诊断
05 condition-aware failure taxonomy
04 controlled single-cue comparison
```

## 当前主结论

USEF-TFGridNet 整体分离能力很强，但 hard condition 会暴露不同类型的局部失败模式。

| Condition | 主要 mismatch 类型 | 关键证据 |
|---|---|---|
| similar-speaker | identity mismatch | ECAPA + SpeechBrain + local waveform preference |
| similar-content v1 | content-attribution mismatch | SSL content gap + local waveform preference |
| similar-content v2 TTS | 更强的 content-attribution mismatch | interferer voice 中的 TTS content stress |

极端失败样本大多可以被 mismatch 解释：

| Condition | Extreme failures | Mismatch-explained |
|---|---:|---:|
| similar-speaker, SI-SNRi < 0 | 323 | 255 |
| similar-content v1, SI-SNRi < 0 | 215 | 207 |
| similar-content v2 TTS, SI-SNRi < 0 | 343 | diagnostic 支持 content-driven failures |

normal/easy USEF reference 已完成。similar-speaker 和 similar-content v1 相对这个内部 reference 没有明显的全局平均 SI-SNRi 下降；目前更强的论点不是“平均性能显著下降”，而是“tail failures 和局部 mismatch attribution”。similar-content v2 TTS 有更清楚的全局下降。

## 当前文件夹结构

| Folder | 内容 |
|---|---|
| `00_overview/` | 研究逻辑、假设、验证缺口、当前状态 |
| `01_data_preparation/` | hard-condition 数据构造和 enrollment leakage 检查 |
| `02_pilot_verifier_sanity/` | clean source 上的 chunk-level verifier sanity |
| `03_pretrained_usef_diagnostic/` | 官方 pretrained USEF-TFGridNet 诊断结果 |
| `04_controlled_single_cue_comparison/` | normal train-100 下 TFMap/Context/USEF single-cue 训练与诊断 |
| `04_next_experiment_plan/` | 之前的 controlled cue comparison 计划矩阵 |
| `05_reports_and_figures/` | 详细报告、case study、taxonomy |
| `99_archive/` | 旧草稿和已被替代的 weak-baseline notes |

## 推荐阅读顺序

1. `00_overview/01_local_mismatch_definition.md`
2. `00_overview/02_pipeline_status.md`
3. `00_overview/03_unified_evidence_chain.md`
4. `01_data_preparation/01_dataset_inventory.md`
5. `02_pilot_verifier_sanity/README.md`
6. `03_pretrained_usef_diagnostic/README.md`
7. `04_controlled_single_cue_comparison/README.md`
8. `05_reports_and_figures/condition_aware_failure_taxonomy/README.md`
9. `05_reports_and_figures/bsrnn_v2_tts_case_attribution/README.md`
10. `04_next_experiment_plan/01_experiment_matrix.md`

## 已完成内容

```text
similar-speaker, similar-content v1, similar-content v2 TTS 数据构造
ECAPA clean-source chunk verifier sanity
SSL content clean-source sanity
USEF similar-speaker full diagnostic
USEF similar-content full diagnostic
USEF similar-content v2 TTS corrected diagnostic
normal/easy USEF reference
SpeechBrain second-verifier check
SSL content diagnostic
condition-aware failure taxonomy
normal train-100 TFMap/Context/USEF single-cue training
100-sample controlled single-cue sanity diagnostic
full controlled single-cue speaker-side diagnostic
full controlled single-cue v2 TTS speaker-side diagnostic
full controlled single-cue v2 TTS SSL content diagnostic
BSRNN v2 TTS case-level attribution
BSRNN v2 TTS ASR/token-level top-case evidence
```

## 当前主要缺口

```text
stronger retrieval-style content verifier calibration 尚未完成
v2 TTS 的 ASR/token-level readable evidence 尚未完成
v2 TTS 的 case-level attribution 尚未完成
```

这些缺口在汇报当前结果时需要明确说明。
