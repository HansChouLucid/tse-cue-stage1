# USEF Similar-Speaker Case Study

## 目的

整理 USEF-TFGridNet 在 full similar-speaker test 上的局部 mismatch 证据。

当前重点不是人工听感，而是用指标判断：

```text
哪些是真失败；
哪些只是 single-verifier disagreement；
confirmed true swap 是否集中在少数 speaker pair；
失败是否持续、是否从开头发生。
```

## 关键文件

| 文件/目录 | 内容 |
|---|---|
| `case_manifest.csv` | 25 个代表 case |
| `01_case_study_interpretation.md` | case study 和第二 verifier 初步解读 |
| `02_full_speechbrain_verifier.md` | full 6000 的 SpeechBrain 复核 |
| `03_confirmed_mismatch_analysis.md` | confirmed mismatch 原因分析 |
| `speechbrain_xvector_verifier/` | 25 个 case 的第二 verifier 结果 |
| `speechbrain_xvector_full/` | full 6000 的第二 verifier 结果 |
| `confirmed_mismatch_analysis/` | pair 集中度、强证据表、分歧复核表 |
| `figures/` | timeline 和全局分布图 |
| `timelines/` | 每个 case 的 chunk-level CSV |

## 当前最重要结论

1. `waveform_global_failure` 和 `true_local_swap` 中的强证据 case 被 SpeechBrain 复核确认。
2. `local_identity_ambiguity` 和 `global_identity_disagreement` 多数是 `single_ecapa_disagreement`，不能直接写成模型 mismatch。
3. confirmed true swap 高度集中在少数 speaker pair。
4. true swap 往往从开头出现，并持续多个 chunk。
5. speaker similarity 不是唯一原因，局部归属错误更像 pair-specific + cue/model 条件共同触发。

## 推荐阅读

1. `03_confirmed_mismatch_analysis.md`
2. `02_full_speechbrain_verifier.md`
3. `01_case_study_interpretation.md`
4. `confirmed_mismatch_analysis/evidence_report/strong_evidence_cases.csv`
5. `confirmed_mismatch_analysis/evidence_report/verifier_disagreement_recheck.csv`
