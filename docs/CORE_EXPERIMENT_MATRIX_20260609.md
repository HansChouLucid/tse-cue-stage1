# Core Experiment Matrix

Date: 2026-06-09

This branch/package keeps only the current Stage-1 mainline. Old similar-content v2 and old similar-content v3 are excluded from the result artifacts and should not be used as evidence.

## Core Scope

Speaker side:

- normal/easy
- hard similar-speaker v2
- hard similar-speaker v3

Content side:

- normal/easy
- quality-first similar-content v3

Backbones:

- Pretrained USEF-TFGridNet
- Pretrained REAL-TSE TFMap+Context
- BSRNN controlled backbone with USEF / TFMap / Context cue

## Dataset Construction

Normal/easy uses the Libri2Mix/LibriMix-style clean two-speaker test condition. It is the sanity baseline: the model should separate target speech under ordinary fixed-enrollment conditions before hard cue-conflict claims are trusted.

Similar-speaker v2 is a speaker-side stress test built by replacing the target enrollment with an utterance from a speaker judged similar to the target speaker. It keeps the mixture natural and stresses speaker identity attribution. v2 has SIR variants such as `sir0db` and `sirm3db`.

Similar-speaker v3 is a harder and more concentrated speaker-side stress test. It uses a smaller set of highly selected similar-speaker pairs and includes `sir0db`, `sirm3db`, and `sirm5db`. Because pair coverage is concentrated, v3 is best treated as a stress test rather than a complete real-world distribution.

Quality-first similar-content v3 is the current content-side mainline. The mixture remains natural Libri2Mix. Only enrollment is replaced: a target-speaker prompt is used to synthesize enrollment speech whose text matches the interferer transcript. The generation was rebuilt after old v2/old v3 quality problems. The current quality-first v3 has 5256 final rows and passed Whisper large-v3 quality checks.

## Coverage Status

| Side | Dataset | Pretrained USEF-TFGridNet | Pretrained TFMap+Context | BSRNN USEF/TFMap/Context |
|---|---|---|---|---|
| Speaker | normal/easy | inference + speaker local diagnostic available | inference available; local sanity not as complete as USEF | traditional metrics and 1k fixed-enroll speaker sanity available |
| Speaker | similar-speaker v2 | inference + speaker local diagnostic available | inference + speaker local diagnostic available | not fully aligned as a three-cue local diagnostic set; treat as a gap |
| Speaker | similar-speaker v3 | inference + speaker local diagnostic available | inference + speaker local diagnostic available | inference + speaker local diagnostic available for three cues |
| Content | normal/easy | inference + content local diagnostic available | inference available; content local diagnostic is less complete than v3 line | inference + SSL/WavLM/HuBERT multi-verifier diagnostic available for three cues |
| Content | quality-first similar-content v3 | inference + content local diagnostic available | inference + content local diagnostic available | inference + content local diagnostic available for three cues |

## Current Interpretation Boundary

The strongest content-side comparison is quality-first similar-content v3 between Pretrained USEF-TFGridNet and Pretrained TFMap+Context. It shows TFMap+Context reduces the content-conflict tail relative to single USEF.

The BSRNN three-cue line is useful for controlled cue comparison, but the backbone is still weaker than the pretrained baselines. BSRNN results should not be interpreted as cue upper bounds until the continuation checkpoints are fully selected and rerun.

Speaker-side v3 is the cleanest hard speaker stress line across all three backbone groups. Speaker-side v2 is complete for pretrained USEF and TFMap+Context, but BSRNN v2 remains a missing aligned diagnostic.

## GitHub Artifact Policy

Included here:

- scripts
- configs
- report snapshot
- small result summaries
- per-utterance CSVs when needed for reproducibility
- data construction manifests and metadata
- logs/curves needed to understand checkpoint choice

Not included here:

- wavs
- generated enrollment audio
- estimated wavs
- checkpoints
- raw LibriSpeech/LibriMix data
- old similar-content v2 / old v3 results

Full local result package:

```text
E:\TSE\latest_core_experiment_artifacts_20260609
E:\TSE\core_stage1_matrix_artifacts_20260609.tar.gz
```

Remote source package:

```text
/data/tse_cue_project/github_stage1_uploads/core_stage1_matrix_artifacts_20260609
/data/tse_cue_project/github_stage1_uploads/core_stage1_matrix_artifacts_20260609.tar.gz
```

