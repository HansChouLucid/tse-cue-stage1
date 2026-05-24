# Stage-1 Dataset Reconstruction Guide

This repository intentionally does not store large wav datasets. It stores the construction logic, configs, small manifests, and experiment results needed to rebuild the data quickly on a high-bandwidth GPU instance.

## What To Rebuild

| Dataset | Previous remote path | Approx size | Why rebuild/save logic instead of data |
|---|---|---:|---|
| similar-speaker 960h | `/data/tse_cue_project/datasets/Libri2Mix_similar_speaker_960h` | 272G | large materialized wavs; can be rebuilt from LibriSpeech + metadata |
| similar-content v1 960h | `/data/tse_cue_project/datasets/Libri2Mix_similar_content_960h` | 293G | large materialized wavs; can be rebuilt from LibriSpeech + metadata |
| similar-content v2 TTS | `/data/tse_cue_project/datasets/Libri2Mix_similar_content_v2_tts` | 2.9G | important but still rebuildable from TTS plan + F5-TTS |
| normal/basic train-100 | `/data/tse_cue_project/datasets/Libri2Mix_normal_metadata_mix` | 17G | small baseline training data; rebuildable |
| WeSep manifests | `/data/tse_cue_project/manifests/...` | 33G total | some audio.json files are large; generation scripts and small v2 manifest are stored here |

## Required Raw Inputs

Place raw data under the standard project root:

```text
/data/tse_cue_project/
  imported_data/
    LibriSpeech960/
      train-clean-100/
      train-clean-360/
      train-other-500/
      dev-clean/
      dev-other/
      test-clean/
      test-other/
    Libri2Mix/
      # Libri2Mix metadata, especially train-clean-100/train-clean-360/dev/test metadata
```

Original LibriSpeech can be downloaded again from OpenSLR. Libri2Mix metadata can also be downloaded/recreated from the public LibriMix recipe. Do not store the tarballs in GitHub.

## Included Reconstruction Tools

The relevant scripts are stored in `dataset_build_tools/`:

| Script | Purpose |
|---|---|
| `build_librimix_style_metadata.py` | Build internal LibriMix-style metadata when needed |
| `build_similar_speaker_from_librimix_metadata.py` | Build similar-speaker metadata from LibriMix metadata |
| `build_similar_speaker_librimix.py` | Materialize similar-speaker mixtures/wavs and WeSep manifests |
| `build_similar_content_from_librimix_metadata.py` | Build similar-content v1 mixtures/wavs/manifests |
| `build_similar_content_v2_tts_plan.py` | Create TTS plan for v2 same-text/similar-content interferers |
| `run_f5tts_plan.py` / `run_f5tts_plan_batch.py` | Generate synthetic interferers from the TTS plan |
| `build_similar_content_v2_dataset_from_tts.py` | Materialize v2 TTS mixtures and metadata after TTS generation |
| `run_scv2_tts_quality_calibration.py` | Check v2 TTS quality / artifacts |
| `prepare_normal_librimix_usef_inputs.py` | Prepare normal/easy reference inputs for USEF diagnostic |

## Recommended Rebuild Order

### 0. Restore code skeleton

```bash
git clone https://github.com/HansChouLucid/tse-cue-stage1.git
cd tse-cue-stage1
bash scripts/restore_skeleton.sh /data/tse_cue_project
```

Then copy `dataset_build_tools/` into your active WeSep tools directory:

```bash
mkdir -p /data/tse_cue_project/code/wesep-real-tse/tools
cp -av dataset_build_tools/*.py /data/tse_cue_project/code/wesep-real-tse/tools/
```

### 1. Download raw LibriSpeech / Libri2Mix metadata

Use high-bandwidth remote download. Keep the standard target paths:

```text
/data/tse_cue_project/imported_data/LibriSpeech960
/data/tse_cue_project/imported_data/Libri2Mix
```

Avoid keeping `/data/tse_cue_project/imports/*.tar.gz` after extraction unless you explicitly want a local cache.

### 2. Rebuild similar-speaker 960h

Expected previous outputs:

```text
/data/tse_cue_project/datasets/Libri2Mix_similar_speaker_960h
/data/tse_cue_project/manifests/librimix_similar_speaker_960h
```

Use:

```bash
cd /data/tse_cue_project/code/wesep-real-tse
source /data/tse_cue_project/scripts/activate_tse.sh
python tools/build_similar_speaker_from_librimix_metadata.py --help
python tools/build_similar_speaker_librimix.py --help
```

The exact CLI may depend on the script version; run `--help` first. Historical build logs are stored under `docs/build_log_tails/` for reference.

### 3. Rebuild similar-content v1 960h

Expected previous outputs:

```text
/data/tse_cue_project/datasets/Libri2Mix_similar_content_960h
/data/tse_cue_project/manifests/librimix_similar_content_960h
```

Use:

```bash
cd /data/tse_cue_project/code/wesep-real-tse
source /data/tse_cue_project/scripts/activate_tse.sh
python tools/build_similar_content_from_librimix_metadata.py --help
```

### 4. Rebuild similar-content v2 TTS

Expected previous outputs:

```text
/data/tse_cue_project/datasets/Libri2Mix_similar_content_v2_tts
/data/tse_cue_project/manifests/librimix_similar_content_v2_tts
/data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep
```

Rebuild flow:

```bash
cd /data/tse_cue_project/code/wesep-real-tse
source /data/tse_cue_project/scripts/activate_tse.sh

# 1. Build TTS plan
python tools/build_similar_content_v2_tts_plan.py --help

# 2. Generate synthetic interferers using F5-TTS
python tools/run_f5tts_plan_batch.py --help

# 3. Materialize mixtures after TTS generation
python tools/build_similar_content_v2_dataset_from_tts.py --help

# 4. Optional: quality calibration
python tools/run_scv2_tts_quality_calibration.py --help
```

The small v2 WeSep manifest from the previous run is stored in:

```text
manifests_small/librimix_similar_content_v2_tts_wesep_clean_test/
```

This helps verify schemas and expected key formats.

## Expected Key Design Decisions

1. Use `aux.scp` / WeSep cue mapping as the authoritative target enrollment direction.
2. Avoid using the current target utterance as enrollment cue when possible.
3. For v2 TTS, the synthetic interferer uses target text spoken in the interferer voice. Therefore ASR evidence checks target-text preservation, while target/interferer attribution still requires SSL content gap + waveform/speaker evidence.
4. Large materialized wav datasets are reproducible and should not be committed to Git.

## Quick Verification After Rebuild

After rebuilding each dataset, verify counts and schema:

```bash
wc -l /data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep/clean/test/raw.list
head -n 2 /data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep/clean/test/samples.jsonl
python -m json.tool /data/tse_cue_project/manifests/librimix_similar_content_v2_tts_wesep/clean/test/cues/audio.json >/dev/null
```

For previous v2 test, expected count was approximately:

```text
5984 samples
```

## Storage Policy

Do not save these in GitHub:

```text
*.wav
*.flac
*.pt
*.tar
*.tar.gz
LibriSpeech / LibriMix raw archives
materialized 960h mixture directories
```

If you later decide to preserve large data, use object storage such as OSS/COS/S3, not GitHub.
