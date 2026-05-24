# USEF Full Similar-Content Diagnostic

## Purpose

This experiment checks whether a strong pretrained fine-grained cue model can produce local content mismatch under a similar-content condition.

```text
similar-speaker: does the output locally drift toward the interferer identity?
similar-content: does the output locally drift toward the interferer content?
```

## Data And Model

| Item | Setting |
|---|---|
| Dataset | Libri2Mix similar-content test |
| Utterances | 5989 |
| Mixture | 2-speaker clean mixture |
| Cue | USEF enrollment cue |
| Model | USEF-TFGridNet pretrained |
| Checkpoint | official USEF-TSE pretrained WSJ0-2mix checkpoint |
| Output sample rate | 8 kHz |

Enrollment cue construction was checked before inference:

```text
self target utterance as enrollment: 0
self source utterance as enrollment: 0
```

## Global Separation Result

| Metric | Value |
|---|---:|
| SI-SNRi mean | 16.84 dB |
| SI-SNRi median | 18.81 dB |
| SI-SNRi < 0 | 215 / 5989 |
| SI-SNRi < 5 | 252 / 5989 |
| SI-SNRi > 15 | 5093 / 5989 |

USEF-TFGridNet is globally strong on similar-content. Most examples are separated well, but about 3.6% are clear failures.

## Local Diagnostic Design

| Evidence | Definition | Role |
|---|---|---|
| content gap | SSL(output chunk, target chunk) minus SSL(output chunk, interferer chunk) | detects whether local content is closer to target or interferer |
| waveform gap | local SI-SDR(output, target) minus local SI-SDR(output, interferer) | checks whether waveform evidence agrees |
| speaker gap | ECAPA(output chunk, target speaker prototype) minus ECAPA(output chunk, interferer prototype) | checks whether content failure is mixed with identity drift |

The main content diagnostic uses 1 s chunks with 0.5 s hop. A negative content gap means the separated output chunk is locally closer to interferer content than target content.

## Main Findings

| Metric | Rate |
|---|---:|
| any content mismatch | 22.83% utterances |
| content mismatch rate > 10% | 14.46% utterances |
| content mismatch rate > 20% | 5.03% utterances |
| any joint content + waveform mismatch | 3.94% utterances |
| joint content + waveform rate > 10% | 3.79% utterances |

The distribution is sparse:

```text
median content mismatch rate: 0
75th percentile: 0
90th percentile: 0.143
```

## Failure-Stratified Result

| Group | Count | Mean content mismatch rate | Mean joint content+wave rate |
|---|---:|---:|---:|
| SI-SNRi < 0 | 215 | 0.881 | 0.860 |
| 0 <= SI-SNRi < 5 | 37 | 0.168 | 0.080 |
| 5 <= SI-SNRi < 15 | 644 | 0.024 | 0.002 |
| SI-SNRi > 15 | 5093 | 0.024 | 0.000 |

When USEF clearly fails globally, the failure is usually strongly aligned with local content moving toward the interferer.

## Cross-Evidence Check

| Metric Pair | Correlation |
|---|---:|
| SI-SNRi vs content mismatch rate | -0.875 |
| SI-SNRi vs joint content+wave rate | -0.920 |
| SI-SNRi vs ECAPA speaker mismatch rate | -0.399 |

Content mismatch explains similar-content failures more directly than speaker mismatch. Speaker mismatch is noisy here: many examples show ECAPA local disagreement while still having high SI-SNRi and low content mismatch.

## Caveat

The SSL content verifier is not an oracle. It assumes that local wav2vec2-style representations preserve phonetic/content similarity well enough for target-vs-interferer comparison.

The strongest evidence should therefore use the joint condition:

```text
content gap < 0
and
local waveform SI-SDR gap < 0
```

## Conclusion

Similar-content exposes a different failure mode from similar-speaker. The current result supports:

```text
local content mismatch exists,
is sparse,
is concentrated in global failure cases,
and is strongest when SSL content gap and waveform gap agree.
```
