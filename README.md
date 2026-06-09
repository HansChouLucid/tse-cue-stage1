# TSE Cue Stage-1 Core Matrix Freeze

This branch/package freezes the current Stage-1 audio-only TSE cue work as of 2026-06-09.

The current formal content-side evidence uses only **quality-first similar-content v3**. Old similar-content v2 and old v3 TTS results are kept out of the result artifacts because they were found to have serious data-quality or construction issues.

The core matrix is:

- Speaker side: normal/easy, hard similar-speaker v2, hard similar-speaker v3.
- Content side: normal/easy, quality-first similar-content v3.
- Backbones: Pretrained USEF-TFGridNet, Pretrained REAL-TSE TFMap+Context, and BSRNN with USEF / TFMap / Context cue.

## What Is Included

- `stage1_report_snapshot/`: latest local HTML report snapshot for advisor-facing reading.
- `scripts/`: current quality-first v3 generation, quality verification, SI-SDR recomputation, and mechanism-probe scripts.
- `result_summaries/`: pruned CSV/JSON summaries for the core experiment matrix. Full wavs and checkpoints are intentionally excluded.
- `wesep_real_tse_stage1_files/`: selected WeSep REAL-TSE files that were edited or used for Stage-1 BSRNN cue training and diagnostics.
- `docs/CORE_EXPERIMENT_MATRIX_20260609.md`: current coverage table, dataset construction summary, and known gaps.
- `ARTIFACTS_MANIFEST.md`: where the large local/remote artifacts live, and which old results must not be cited.
- `CLEANUP_REVIEW_20260609.md`: cleanup candidates that should be deleted or moved only after explicit approval.

## Current Evidence Boundary

Use as current mainline:

- Pretrained USEF-TFGridNet on normal/easy, similar-speaker v2/v3, and quality-first similar-content v3.
- Pretrained REAL-TSE TFMap+Context on normal/easy, similar-speaker v2/v3, and quality-first similar-content v3.
- BSRNN controlled cue comparison as a weak/continuation backbone line; 18-epoch results should not be treated as final cue upper bounds.
- quality-first similar-content v3 quality validation:
  - final rows: 5256
  - Whisper large-v3 WER mean: 0.033
  - CER mean: 0.015
  - token F1 mean: 0.971
  - ASR pass rate: 99.98%

Do not cite as formal evidence:

- similar-content v2 TTS
- old similar-content v3 TTS enrollment-conflict
- SCV2TTS diagnostic/case-study tables
- recovered GitHub v2 TTS artifacts

Known gap:

- BSRNN three-cue speaker-side v2 local diagnostic is not fully aligned with the v3 speaker-side diagnostic set. It is recorded as a gap rather than filled with unrelated old results.

## GitHub Upload Plan

Recommended branch name:

```bash
stage1-quality-v3-freeze-20260609
```

Recommended commit message:

```bash
Freeze stage1 quality-first v3 reports and scripts
```

This package is intentionally small enough for GitHub. Large data, checkpoints, generated wavs, and full diagnostics should stay local/remote or be moved to object storage.
