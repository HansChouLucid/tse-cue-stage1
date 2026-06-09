# Artifact Manifest

Date: 2026-06-09

This file defines the artifact boundary for the current Stage-1 TSE cue work.

## Local Current Assets

Current formal report:

```text
E:\TSE\audio-only TSE cue\deliverables\LATEST_audio_only_tse_cue_report_20260608
```

Current GitHub staging package:

```text
E:\TSE\github_stage1_quality_v3_freeze_20260609
```

Local code/source references:

```text
E:\TSE\wesep-real-tse
E:\TSE\audio-only TSE cue\deliverables\LATEST_audio_only_tse_cue_report_20260608\scripts
```

Local large data that should not be uploaded to GitHub:

```text
E:\TSE\data
E:\TSE\transfer_staging
E:\TSE\pretrained_backbones
```

## Remote Current Assets

Remote host used for audit:

```text
root@223.109.239.36 -p 16236
```

Remote project root:

```text
/data/tse_cue_project
```

Current quality-first similar-content v3 full pipeline:

```text
/data/tse_cue_project/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
/data/tse_cue_project/manifests/librimix_similar_content_v3_tts_enroll_conflict_quality_first_20260609
```

Current BSRNN training/checkpoint area:

```text
/data/tse_cue_project/experiments/normal_basic_single_cue
/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608
```

Current pretrained baseline area:

```text
/data/tse_cue_project/pretrained/real_tse_baselines
```

Current similar-speaker datasets:

```text
/data/tse_cue_project/datasets/librimix_similar_speaker_hard_v2_*
/data/tse_cue_project/datasets/librimix_similar_speaker_hard_v3_*
/data/tse_cue_project/datasets/Libri2Mix_similar_speaker_clean
```

## Obsolete / Do Not Cite

These artifacts should not be used as formal evidence because later checks found serious data-quality or construction problems.

Remote obsolete folder:

```text
/data/tse_cue_project/obsolete_similar_content_20260609
```

Remote old content still visible outside obsolete folders and should be ignored or moved after approval:

```text
/data/tse_cue_project/diagnostics/scv2_*
/data/tse_cue_project/diagnostics/*scv2_tts*
/data/tse_cue_project/diagnostics/*similar_content*v2*
/data/tse_cue_project/logs/scv2_*
/data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep
/data/tse_cue_project/repos/tse-cue-stage1/manifests_small/librimix_similar_content_v2_tts_wesep_clean_test
```

Local obsolete/problem content:

```text
E:\TSE\recovered_v2tts_from_github
E:\TSE\github_tse_cue_stage1_main\tse-cue-stage1-main
E:\TSE\audio-only TSE cue\experiment_pipeline\*similar_content_v2*
E:\TSE\tools\*scv2*
E:\TSE\tools\*v2_tts*
```

Important nuance: old scripts can be retained for forensic traceability, but they should not be placed in the current mainline GitHub branch unless clearly archived and marked obsolete.

## Storage Recommendation

Use GitHub for:

- scripts
- configs
- report snapshots
- small CSV/JSON summaries
- artifact manifests

Do not use GitHub for:

- wav files
- LibriMix/LibriSpeech data
- checkpoints
- full generated diagnostics
- ASR model caches

For large artifacts, prefer one of:

- remote `/data/tse_cue_project` plus this manifest
- cloud object storage such as OSS/S3/R2
- Hugging Face Datasets for publishable manifests and small derived metadata
- Hugging Face Hub or Git LFS only for selected small checkpoints, not raw full experiment dumps

