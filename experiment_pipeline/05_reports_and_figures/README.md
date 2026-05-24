# Reports and Figures

这里存放可以直接用于阅读、汇报和人工复核的结果整理。

## 当前已有结果

```text
usef_similar_speaker_cases/
```

内容：

```text
case_manifest.csv
global_identity_disagreement_checklist.csv
figures/
timelines/
README.md
```

这批文件来自：

```text
USEF-TFGridNet pretrained
similar_speaker full test
taxonomy v2
```

## 推荐先看

1. `usef_similar_speaker_cases/README.md`
2. `usef_similar_speaker_cases/case_manifest.csv`
3. `usef_similar_speaker_cases/figures/scatter_sisnri_vs_mismatch_rate_1s.png`
4. `usef_similar_speaker_cases/figures/distribution_mismatch_rate.png`
5. `usef_similar_speaker_cases/figures/` 中各类 case 的 timeline 图

## 图表含义

每个 timeline 图：

```text
speaker_gap(t): 局部 speaker verifier 更接近 target 还是 interferer
local_sisdr_gap(t): 局部 waveform 更接近 target 还是 interferer
```

阴影：

```text
黄色：speaker_gap < 0, local_sisdr_gap > 0
红色：speaker_gap < 0, local_sisdr_gap < 0
```

黄色对应 identity ambiguity，红色对应更强的 true local swap。

## 下一步要补的报告

1. 人工听感记录表。
2. global identity disagreement 的第二 verifier 复核结果。
3. similar-content diagnostic 的 content-level timeline。
