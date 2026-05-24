# TSE Cue Stage-1 Recovery Repository

This repository stores small, critical artifacts for the audio-only TSE cue stage-1 mismatch study. It is designed for fast recovery after renting a new GPU instance.

It intentionally does **not** contain large wav datasets, LibriSpeech/LibriMix archives, full conda environments, or materialized 960h mixtures.

## What Is Included

- `experiment_pipeline/`: local research pipeline notes and reports.
- `stage1_html_report/`: browser-friendly HTML summary package.
- `code_tools/`: diagnostic / inference helper scripts used in the experiments.
- `configs/`: WeSep experiment configs.
- `results/`: aggregate reports, SSL summaries, case attribution CSV/JSON.
- `manifests_small/`: small v2 TTS WeSep manifest files.
- `remote_inventory_20260525.md`: storage snapshot before pausing the instance.

## What Is Not Included

- `/data/tse_cue_project/datasets/Libri2Mix_similar_speaker_960h` (~272G)
- `/data/tse_cue_project/datasets/Libri2Mix_similar_content_960h` (~293G)
- `/data/tse_cue_project/imports` (~62G)
- full LibriSpeech / LibriMix audio archives
- complete conda environments

## Quick Recovery Sketch

```bash
git clone https://github.com/HansChouLucid/tse-cue-stage1.git
cd tse-cue-stage1
# Read experiment_pipeline/README.md and stage1_html_report/index.html first.
```

For a full second-stage run, rebuild or re-download the large datasets on the remote GPU instance, then copy these configs/scripts/results into `/data/tse_cue_project`.

## Stage-1 Conclusion

The current evidence supports that fine-grained cue TSE can produce local, finite, diagnosable target/interferer mismatch under hard conditions. Similar-speaker mainly exposes speaker identity mismatch; similar-content v2 TTS mainly exposes content attribution mismatch. Controlled BSRNN comparison shows USEF is more stable than TFMap/Context in the current setup.

## Dataset Reconstruction

Large constructed datasets are not stored here. The reconstruction plan and scripts are included instead:

- `docs/DATA_RECONSTRUCTION.md`
- `dataset_build_tools/`
- `scripts/rebuild_stage1_datasets.sh`

After raw LibriSpeech / Libri2Mix metadata is available on a new remote instance, run:

```bash
bash scripts/rebuild_stage1_datasets.sh /data/tse_cue_project
```
