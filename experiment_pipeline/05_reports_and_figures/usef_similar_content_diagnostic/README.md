# USEF Similar-Content Diagnostic Report

## What Was Run

Full test diagnostics for USEF-TFGridNet pretrained on two similar-content conditions:

```text
v1: natural LibriSpeech text overlap / similarity
v2: TTS interferer is prompted to speak the target transcript in the interferer voice
```

Remote outputs:

```text
/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_content_test_full/usef_tfgridnet_wsj0_2mix
/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet/similar_content_v2_tts_test_full/usef_tfgridnet_wsj0_2mix
```

Main generated files:

| File | Purpose |
|---|---|
| `inference_summary.csv` | global SI-SNR / SI-SNRi |
| `local_content_ssl/ssl_content_summary_1s.json` | SSL content mismatch aggregate |
| `local_content_ssl/ssl_content_per_utterance_1s.csv` | utterance-level content summary |
| `local_mismatch_ecapa/per_utterance_summary.csv` | speaker-side local diagnostic |
| `similar_content_condensed_analysis.json` | combined content / waveform / speaker analysis |

## Short Conclusion

USEF-TFGridNet is globally strong on v1, but v2 exposes a stronger content-stress failure mode.

```text
v1 SI-SNRi < 0 examples: mean content mismatch rate 0.881
v2 SI-SNRi < 0 examples: mean content mismatch rate 0.919
```

So when the model fails badly, it is usually not just mild distortion. The output is locally, and often extensively, closer to the interferer content.

## Key Numbers

| Metric | v1 | v2 TTS |
|---|---:|---:|
| utterances | 5989 | 5984 |
| SI-SNRi mean | 16.84 dB | 13.18 dB |
| SI-SNRi median | 18.81 dB | 15.81 dB |
| SI-SNRi < 0 | 3.59% | 5.73% |
| any content mismatch | 22.83% | 20.85% |
| content mismatch rate > 20% | 5.03% | 7.83% |
| joint content+waveform rate > 10% | 3.79% | 5.99% |

## How To Read The Evidence

Content mismatch:

```text
SSL(output chunk, target chunk) < SSL(output chunk, interferer chunk)
```

Waveform confirmation:

```text
local SI-SDR(output, target) < local SI-SDR(output, interferer)
```

The joint metric is stricter than content-only and is the better number to cite when making a strong claim.

## Speaker Diagnostic

ECAPA speaker mismatch is frequent:

```text
any ECAPA speaker mismatch: 62.75%
ECAPA speaker mismatch rate > 20%: 33.88%
```

This should not be interpreted as true speaker swapping by itself. A large subset has high SI-SNRi and low content mismatch. In similar-content, ECAPA is useful as a side check, not the primary verifier.

## Research Interpretation

| Condition | Primary mismatch evidence |
|---|---|
| similar-speaker | multi-verifier identity + waveform confirmation |
| similar-content | SSL content gap + local waveform confirmation |

For similar-content, the current evidence is best framed as content-attribution mismatch rather than pure speaker identity mismatch.

v2 adds a stronger statement:

```text
TTS content-stress construction can turn the content condition from a tail-only diagnostic
into a broad degradation condition for USEF.
```

## v2 TTS Quality Calibration

The v2 quality checks reduce some artifact concerns:

| Check | Result |
|---|---:|
| synthetic prefers interferer voice | 99.16% |
| low-SI-SNRi voice-preservation rate | 99.13% |
| good-SI-SNRi voice-preservation rate | 99.15% |
| final interferer / target RMS ratio | 1.00 mean |
| content mismatch rate correlation with SI-SNRi | -0.894 |
| joint content + waveform correlation with SI-SNRi | -0.930 |
| speaker voice gap correlation with SI-SNRi | 0.011 |
| duration ratio correlation with SI-SNRi | -0.033 |

This means the v2 failures are much more aligned with local content/waveform mismatch than with obvious TTS voice, energy, duration, or clipping artifacts.

The important caveat is text correctness. Wav2vec2 ASR sanity on a sample gives high WER for synthetic interferers:

| Group | Synthetic WER | Target-clean WER |
|---|---:|---:|
| low SI-SNRi | 0.966 | 0.164 |
| good SI-SNRi | 1.003 | 0.225 |
| random | 0.999 | 0.181 |

So v2 should not yet be presented as a clean strict same-text benchmark. It is better described as a strong TTS content-stress set. Since synthetic WER is similarly high in low and good samples, the current evidence still supports local content mismatch as the reason for USEF's failed subset.
