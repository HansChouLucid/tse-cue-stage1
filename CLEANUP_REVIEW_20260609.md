# Cleanup Review Before Deletion

No deletion has been performed for the following candidates. They are listed for approval.

## Local Safe Deletion Candidates

These are mostly duplicate archives or obsolete bad-result recoveries.

| Candidate | Approx. size | Reason |
|---|---:|---|
| `E:\TSE\transfer_staging\librispeech_clean100_dev_test_librimix_meta.tar` | 4.30 GB | transfer tar; extracted/source data already exist elsewhere |
| `E:\TSE\data\LibriMixData\wham_noise.zip` | 16.9 GB | compressed source archive; keep only if rebuilding WHAM noise locally is expected |
| `E:\TSE\data\LibriMixData\train-clean-100.tar.gz` | 3.47 GB | compressed LibriSpeech archive; extracted LibriSpeech exists locally |
| `E:\TSE\data\LibriMixData\dev-clean.tar.gz` and `test-clean.tar.gz` | 0.64 GB | compressed LibriSpeech archives; extracted data exists locally |
| `E:\TSE\recovered_v2tts_from_github` | 0.06 GB | recovered old similar-content v2 artifacts; obsolete and should not be cited |
| `E:\TSE\tools\__pycache__` | tiny | generated Python cache |

Estimated local space recoverable: about 25 GB.

## Local Move-To-Archive Candidates

These are not necessarily useless, but should not be mistaken for the current report/code line.

| Candidate | Action | Reason |
|---|---|---|
| `E:\TSE\github_tse_cue_stage1_main\tse-cue-stage1-main` | move to `_archive` or delete after approval | stale clone-like folder with old v2 content |
| `E:\TSE\audio-only TSE cue\tmp_github_upload` | move to `_archive` or delete after approval | old GitHub upload staging |
| `E:\TSE\audio-only TSE cue\experiment_pipeline` old v2 documents | keep under `99_archive` or exclude from GitHub | process history only, not current evidence |
| `E:\TSE\tools\*scv2*` and `E:\TSE\tools\*v2_tts*` | archive or exclude from main branch | old similar-content v2 tooling |

## Remote Move/Delete Candidates

Remote disk is not urgent: `/data` has about 520 GB available. I recommend moving before deleting.

| Candidate | Suggested action | Reason |
|---|---|---|
| `/data/tse_cue_project/diagnostics/scv2_*` | move under `/data/tse_cue_project/obsolete_similar_content_20260609/diagnostics/` | old v2 diagnostics outside obsolete folder |
| `/data/tse_cue_project/diagnostics/*scv2_tts*` | move under obsolete folder | old v2 diagnostics |
| `/data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep` | move under obsolete folder | old v2 manifest |
| `/data/tse_cue_project/logs/scv2_*` | move under obsolete folder | old v2 logs |
| per-epoch BSRNN checkpoints outside `selected_checkpoints_20260608` | keep for now, later prune to selected checkpoints plus logs | checkpoint selection may still need audit |

## Keep

Do not delete or move:

```text
E:\TSE\audio-only TSE cue\deliverables\LATEST_audio_only_tse_cue_report_20260608
E:\TSE\github_stage1_quality_v3_freeze_20260609
E:\TSE\wesep-real-tse
/data/tse_cue_project/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
/data/tse_cue_project/manifests/librimix_similar_content_v3_tts_enroll_conflict_quality_first_20260609
/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608
```

