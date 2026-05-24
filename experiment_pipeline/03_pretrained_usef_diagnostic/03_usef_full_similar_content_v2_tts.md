# USEF Full Similar-Content v2 TTS Diagnostic

## Purpose

This experiment tests a harder content-stress condition:

```text
the interferer voice is prompted to speak the target transcript
```

The goal is to check whether USEF-TFGridNet becomes more fragile when target and interferer content are much closer than natural LibriSpeech overlap.

## Data And Model

| Item | Setting |
|---|---|
| Dataset | Libri2Mix similar-content v2 TTS test |
| Utterances | 5984 |
| Target | original LibriSpeech target utterance |
| Interferer | F5-TTS synthetic speech in interferer voice, using target transcript |
| SIR | 0 dB |
| Cue | target-speaker enrollment cue |
| Model | public pretrained USEF-TFGridNet |
| Checkpoint | USEF-TSE WSJ0-2mix checkpoint |
| Output sample rate | 8 kHz |

Enrollment correction:

```text
The first v2 build accidentally used the interferer prompt as enrollment.
That run was archived as wrong_aux and should not be used as a research result.
The corrected build uses target-speaker enrollment and avoids the current target utterance.
```

## Global Separation Result

| Condition | Mean | Median | p10 | SI-SNRi < 0 | SI-SNRi < 5 |
|---|---:|---:|---:|---:|---:|
| normal/easy reference | 15.90 dB | 18.69 dB | 12.03 dB | 310 / 6000 | 347 / 6000 |
| similar-content v1 | 16.84 dB | 18.81 dB | 13.53 dB | 215 / 5989 | 252 / 5989 |
| similar-content v2 TTS | 13.18 dB | 15.81 dB | 9.39 dB | 343 / 5984 | 397 / 5984 |

Relative result:

```text
v2 is 2.72 dB below normal/easy in mean SI-SNRi.
v2 is 3.66 dB below similar-content v1 in mean SI-SNRi.
```

This is the first content condition here that shows a clear broad degradation for USEF, not only tail failures.

## Local Content Diagnostic

The diagnostic uses 1 s chunks with 0.5 s hop.

Content mismatch:

```text
SSL(output chunk, target chunk) < SSL(output chunk, interferer chunk)
```

Waveform confirmation:

```text
local SI-SDR(output, target) < local SI-SDR(output, interferer)
```

Key results:

| Metric | Value |
|---|---:|
| utterances analyzed | 5980 |
| chunks analyzed | 54083 |
| any content mismatch | 20.85% |
| content mismatch rate > 10% | 13.49% |
| content mismatch rate > 20% | 7.83% |
| any joint content + waveform mismatch | 6.14% |
| joint content + waveform rate > 10% | 5.99% |

Failure split:

| Group | Count | Mean content mismatch rate |
|---|---:|---:|
| SI-SNRi < 0 | 343 | 0.919 |
| SI-SNRi > 15 | 3530 | 0.017 |

## Interpretation

The result supports the user's hypothesis more strongly than v1:

```text
when the interferer says the target content in another voice,
USEF is more likely to produce local content-attribution mismatch,
and this mismatch is highly concentrated in the global failure cases.
```

This does not yet prove the whole degradation is caused only by fine-grained cue mismatch. v2 uses TTS, so a complete claim needs TTS quality calibration.

## TTS Quality Calibration

Quality calibration was run on all 5984 samples for audio statistics and speaker identity, plus an ASR text-sanity sample.

Remote outputs:

```text
/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_content_v2_tts_test_full/tts_quality_calibration
```

Audio and speaker checks:

| Check | Result | Interpretation |
|---|---:|---|
| synthetic prefers interferer voice | 99.16% | TTS voice conditioning mostly works |
| low-SI-SNRi prefers interferer voice | 99.13% | failures are not caused by losing interferer voice |
| good-SI-SNRi prefers interferer voice | 99.15% | voice preservation is similar in good and bad samples |
| raw synth / target duration ratio, mean | 1.03 | duration is usually close |
| final interferer / target RMS ratio, mean | 1.00 | 0 dB SIR scaling is correct |
| mix clipping present | 161 / 5984 | small number; not concentrated in failures |
| synth clipping present | 199 / 5984 | small number; not concentrated in failures |

Failure relation:

| Metric | Correlation with SI-SNRi |
|---|---:|
| content mismatch rate | -0.894 |
| joint content + waveform rate | -0.930 |
| waveform interferer rate | -0.932 |
| raw synth / target duration ratio | -0.033 |
| mix clipping rate | 0.026 |
| synthetic clipping rate | 0.013 |
| speaker voice gap | 0.011 |

Interpretation:

```text
The USEF failures align strongly with local content/waveform mismatch,
not with obvious duration, energy, clipping, or voice-preservation artifacts.
```

ASR text sanity:

| Group | Synthetic WER | Target-clean WER |
|---|---:|---:|
| low SI-SNRi sample | 0.966 | 0.164 |
| good SI-SNRi sample | 1.003 | 0.225 |
| random sample | 0.999 | 0.181 |

Interpretation:

```text
The synthetic interferers are not clean ASR-equivalent copies of the target transcript.
Therefore v2 should be described as a strong TTS content-stress set, not a clean strict same-text benchmark.
However, synthetic WER is similarly high in low- and high-SI-SNRi groups, so it does not explain why USEF fails on the low-SI-SNRi subset.
```

## Required Follow-Up

1. Improve v2 with stronger TTS or a same-text multi-speaker source if final claims require clean transcript control.
2. Run the same v2 condition for TFMap-only and Context-only cues.
3. Keep v2 conclusions phrased as content-stress evidence unless text quality is improved.
