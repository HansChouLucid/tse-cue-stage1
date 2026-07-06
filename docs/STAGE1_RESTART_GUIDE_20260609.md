# Stage-1 TSE Cue Restart Guide

Date: 2026-06-09

This guide is the restart entry point for the current Stage-1 audio-only TSE cue work. It is written for a future experimenter or AI agent who needs to understand what is valid, what has been saved locally, how the main metrics are defined, and how to restart experiments after the remote machine is cleared.

## Current Valid Scope

The current mainline studies whether fine-grained audio-only cues can suffer from local target/interferer attribution drift under controlled speaker-side and content-side stress tests.

Use only the current mainline datasets:

- Speaker side normal/easy baseline.
- Speaker side similar-speaker v2.
- Speaker side similar-speaker v3.
- Content side normal/easy baseline.
- Content side quality-first similar-content v3.

Use only the current mainline backbone groups:

- Pretrained USEF-TFGridNet.
- Pretrained REAL-TSE TFMap+Context.
- BSRNN controlled backbone with USEF / TFMap / Context single cue.

Do not cite old similar-content v2 or old similar-content v3 TTS results. They were invalidated by data-quality and construction checks.

## Local Assets To Keep

Current constructed datasets:

```text
E:\TSE\current_constructed_datasets_20260609
E:\TSE\current_constructed_datasets_20260609.tar.gz
```

This contains:

- `speaker_side/datasets/librimix_similar_speaker_hard_v2_sir0db`
- `speaker_side/datasets/librimix_similar_speaker_hard_v2_sirm3db`
- `speaker_side/datasets/librimix_similar_speaker_hard_v3_sir0db`
- `speaker_side/datasets/librimix_similar_speaker_hard_v3_sirm3db`
- `speaker_side/datasets/librimix_similar_speaker_hard_v3_sirm5db`
- `content_side/similar_content_v3_quality_first/final_enrollment`
- `content_side/similar_content_v3_quality_first/manifest`
- `content_side/similar_content_v3_quality_first/quality_manifest.csv`
- `content_side/similar_content_v3_quality_first/whisper_large_v3_quality`

Current BSRNN checkpoints:

```text
E:\TSE\bsrnn_selected_checkpoints_freeze_20260609
```

This contains best and averaged BSRNN checkpoints for:

- `usef_only_bsrnn`
- `tfmap_only_bsrnn`
- `context_only_bsrnn`

Selection metadata:

```text
E:\TSE\bsrnn_selected_checkpoints_freeze_20260609\BSRNN_SELECTED_CHECKPOINTS_FREEZE_20260609.json
```

Current code and summaries:

```text
E:\TSE\github_stage1_quality_v3_freeze_20260609
E:\TSE\latest_core_experiment_artifacts_20260609
E:\TSE\audio-only TSE cue\deliverables\LATEST_audio_only_tse_cue_report_20260608
```

GitHub branch:

```text
stage1-quality-v3-freeze-20260609
commit b98519e334df639554c5cfd7848190d4536d867d
```

## Assets That Can Be Re-Downloaded

The following are not required to be preserved locally:

- LibriSpeech / LibriMix raw data.
- Pretrained USEF-TFGridNet checkpoint.
- Pretrained REAL-TSE TFMap+Context checkpoint.
- Whisper / WavLM / HuBERT / Wav2Vec2 / F5-TTS model caches.
- Full estimated wav outputs.
- Full diagnostics intermediate wavs.

These can be rebuilt or re-downloaded if the scripts and manifests are available.

## Dataset Construction Summary

Normal/easy is the ordinary clean LibriMix/Libri2Mix-style two-speaker fixed-enrollment test condition. It is the sanity baseline: if a backbone is weak here, hard-set degradation should not be over-interpreted as a cue-specific phenomenon.

Similar-speaker v2 replaces the target enrollment with an utterance from a similar speaker. The mixture remains natural. It includes SIR variants such as `sir0db` and `sirm3db`.

Similar-speaker v3 is a stronger speaker-side stress test. It uses a smaller, more concentrated set of selected similar-speaker pairs and includes `sir0db`, `sirm3db`, and `sirm5db`. Because pair diversity is concentrated, v3 should be interpreted as a stress test, not a full real-world distribution.

Quality-first similar-content v3 keeps the mixture natural and replaces only the enrollment. The synthesized enrollment keeps the target-speaker voice but uses text aligned with the interferer-side content conflict. This version replaced old v2/old v3 after quality problems were found. The current package includes final enrollment audio, manifests, and Whisper large-v3 quality metadata.

## Core Metrics

Traditional global metrics:

- `Output SI-SDR`: scale-invariant signal-to-distortion ratio between model output and target clean source. Higher is better.
- `SI-SDRi`: output SI-SDR minus mixture SI-SDR. Higher is better. This measures how much the model improves over the input mixture.
- `Output SI-SNR`: scale-invariant signal-to-noise ratio between model output and target clean source. Higher is better.
- `SI-SNRi`: output SI-SNR minus mixture SI-SNR. Higher is better. This is the traditional TSE/separation improvement metric.

Local waveform attribution metric:

- `Target-interferer SI-SDR gap`: local SI-SDR(output, target chunk) minus SI-SDR(output, interferer chunk). Positive means the local output is closer to the target than the interferer. Negative means the local waveform is closer to the interferer and is a strong local drift signal.

Speaker-side local verifier metrics:

- `Speaker mismatch`: a speaker verifier judges a local output chunk closer to the interferer speaker than to the target speaker.
- `Speaker confirmed drift`: speaker mismatch that is also supported by waveform evidence, usually via negative or weak target-interferer SI-SDR gap. This is stricter than raw verifier mismatch.
- `Multi-verifier speaker mismatch`: mismatch agreed or summarized across ECAPA and x-vector style verifiers.
- `Any confirmed drift case`: utterance-level rate where at least one local chunk has confirmed drift.
- `Persistent confirmed drift case`: utterance-level rate where confirmed drift occupies a meaningful fraction of chunks, for example at or above a threshold such as 20%.

Content-side local verifier metrics:

- `Content mismatch`: a content SSL verifier judges a local output chunk closer to the interferer/content-conflict reference than the target reference.
- `Content confirmed drift`: content mismatch that is supported by waveform evidence via target-interferer SI-SDR gap.
- `2-verifier content drift`: content drift supported by two content verifiers, such as WavLM and HuBERT.
- `Multi-verifier content drift`: content drift summarized across SSL/WavLM/HuBERT-style checks.
- `Any content drift case`: utterance-level rate where at least one local chunk has content drift.
- `Persistent content drift case`: utterance-level rate where content drift occupies a meaningful fraction of chunks.

Verifier sanity metrics:

- `AUC`: verifier discrimination ability on clean/reference chunks. AUC near 1 means the verifier reliably separates target-like from interferer-like examples; AUC near 0.5 means weak discrimination.
- `Verifier instability`: local mismatch that appears even on clean-source sanity checks. High instability means the verifier itself is unreliable and local mismatch results should be interpreted cautiously.

Mechanism-level probe metrics:

- `Internal target-interferer margin`: similarity(cue representation, target representation) minus similarity(cue representation, interferer representation), computed on aligned local frames/chunks when available. Positive means the cue representation is more target-aligned; negative means cue ambiguity favors the interferer.
- `Negative-margin frame rate`: fraction of local cue frames where internal margin is negative.
- `Attention entropy`: how diffuse the attention distribution is. Higher entropy means less concentrated attention.
- `Attention peak`: maximum attention probability. Lower peak often means more diffuse attention.

Mechanism probes are not direct proof of output error. They show cue-level ambiguity. Output local diagnostics determine whether separator behavior corrects or converts that ambiguity into waveform/content/speaker drift.

## How To Interpret SI-SDRi With Local Drift

Use SI-SDRi and local drift together:

- High SI-SDRi + high drift: most research-relevant. Global quality is good, but local attribution risk is hidden by the average metric.
- Low SI-SDRi + high drift: failure-tail samples. Local diagnostics help decide whether the failure is ordinary separation failure or target/interferer attribution drift.
- Low SI-SDRi + low drift: ordinary separation difficulty or undertrained backbone.
- High SI-SDRi + low drift: stable success.

Do not use SI-SDRi alone to claim there is no local mismatch. SI-SDRi is utterance-level and can hide short but meaningful target/interferer swaps.

## Valid Result Interpretation Boundary

Pretrained USEF-TFGridNet and pretrained TFMap+Context are the strongest baselines for interpreting model behavior. BSRNN is useful for controlled cue comparison because only the cue changes, but its backbone quality is weaker than the pretrained baselines.

For content-side quality-first v3, the most important comparison is between pretrained USEF-TFGridNet and pretrained TFMap+Context, plus BSRNN three-cue diagnostic as controlled support.

For speaker-side stress, v3 is the cleanest hard-set line across backbones. v2 is useful but less aligned for BSRNN three-cue local diagnostics.

## Fast Restart Procedure

1. Clone or restore code from GitHub branch:

```text
stage1-quality-v3-freeze-20260609
```

2. Restore local data from:

```text
E:\TSE\current_constructed_datasets_20260609
```

3. Restore BSRNN checkpoints from:

```text
E:\TSE\bsrnn_selected_checkpoints_freeze_20260609
```

4. Re-download pretrained checkpoints if needed:

- USEF-TFGridNet.
- REAL-TSE TFMap+Context.

5. Re-run inference by side:

- Speaker side: normal/easy, similar-speaker v2, similar-speaker v3.
- Content side: normal/easy, quality-first similar-content v3.

6. Re-run diagnostics:

- Speaker local diagnostic with ECAPA/x-vector multi-verifier.
- Content local diagnostic with SSL/WavLM/HuBERT multi-verifier.
- Traditional SI-SDR/SI-SDRi and SI-SNR/SI-SNRi.

7. Regenerate HTML/report tables from summary CSV/JSON outputs.

## Do Not Use

Do not use or cite:

- Old similar-content v2 TTS results.
- Old similar-content v3 before quality-first rebuild.
- USEF-SepFormer exploratory results.
- Full old case-study outputs derived from invalid similar-content v2/v3.

## Minimal Survival Set Before Remote Clear

Before clearing the remote machine, confirm these exist locally:

```text
E:\TSE\current_constructed_datasets_20260609
E:\TSE\current_constructed_datasets_20260609.tar.gz
E:\TSE\bsrnn_selected_checkpoints_freeze_20260609
E:\TSE\github_stage1_quality_v3_freeze_20260609
E:\TSE\latest_core_experiment_artifacts_20260609
E:\TSE\audio-only TSE cue\deliverables\LATEST_audio_only_tse_cue_report_20260608
```

If these are present, the remote can be cleared with acceptable risk.
