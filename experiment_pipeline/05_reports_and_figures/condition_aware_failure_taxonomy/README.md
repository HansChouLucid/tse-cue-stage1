# Condition-Aware Failure Taxonomy

## Purpose

This report answers two questions:

```text
1. Are extreme negative samples caused by mismatch?
2. Should relative SI-SNRi degradation also count as failure?
```

The answer is condition-dependent. Similar-speaker and similar-content need different primary mismatch verifiers.

## Primary Evidence By Condition

| Condition | Primary mismatch evidence | Reason |
|---|---|---|
| similar-speaker | multi-verifier identity true-swap + waveform preference | the hard factor is speaker identity similarity |
| similar-content | SSL content mismatch + waveform preference | the hard factor is content attribution |

This avoids using ECAPA speaker disagreement as a universal verifier.

## Reference SI-SNRi Distribution

| Condition | p10 | median | p75 |
|---|---:|---:|---:|
| similar-speaker | 10.99 dB | 18.49 dB | 20.64 dB |
| similar-content | 13.53 dB | 18.81 dB | 20.86 dB |

The p10 threshold is used as a relative tail marker. This catches samples that are not necessarily negative but are weak relative to the condition.

## Extreme Failure Attribution

### Similar-Speaker

| Group | Count |
|---|---:|
| SI-SNRi < 0 | 323 |
| mismatch-explained extreme failure | 255 |
| non-mismatch or unconfirmed failure | 66 |
| identity disagreement without waveform swap | 2 |

Interpretation:

```text
About 79% of negative similar-speaker samples are explained by confirmed identity mismatch.
The remaining 21% need separate treatment: collapse, residual mixture, artifacts, or verifier miss.
```

### Similar-Content

| Group | Count |
|---|---:|
| SI-SNRi < 0 | 215 |
| mismatch-explained extreme failure | 207 |
| content-only, waveform-unconfirmed | 4 |
| non-mismatch or unconfirmed failure | 4 |

Interpretation:

```text
About 96% of negative similar-content samples are explained by joint content + waveform mismatch.
This is a very strong signal that extreme similar-content failure usually means content attribution failure.
```

## Weak Or Failed Samples

Here weak means SI-SNRi < 5 dB.

| Condition | Count | Mismatch-explained | Non-mismatch / unconfirmed |
|---|---:|---:|---:|
| similar-speaker | 404 | 271 | 122 |
| similar-content | 252 | 220 | 24 |

Interpretation:

```text
Mismatch explains most weak/failing samples, but the unexplained portion grows when the threshold is relaxed from <0 to <5.
```

## Relative Tail Samples

Relative tail means the low end of each condition:

```text
SI-SNRi <= condition p10
plus absolute failure and weak bands
```

| Condition | Tail count | Main mismatch-explained count |
|---|---:|---:|
| similar-speaker | 600 | 271 |
| similar-content | 599 | 220 |

Interpretation:

```text
Relative degradation is broader than confirmed mismatch.
It captures partial degradation, verifier instability, and normal difficult cases.
It should be reported, but not all relative-tail samples should be called mismatch.
```

## ECAPA Instability

ECAPA disagreement appears frequently, especially when used as a local 1 s chunk verifier.

| Condition | Instability-like category |
|---|---:|
| similar-speaker single-verifier instability | 1976 |
| similar-content speaker-verifier instability | 1596 |

This does not invalidate the project. It shows why local mismatch diagnosis cannot rely on a single verifier.

Recommended framing:

```text
ECAPA alone = weak evidence
multi-verifier + waveform = strong identity evidence
SSL content + waveform = strong content evidence
```

## Working Taxonomy

| Failure Cause | Meaning |
|---|---|
| mismatch-explained extreme failure | global failure with primary mismatch evidence |
| confirmed local identity/content mismatch | local mismatch exists but global metric is not necessarily failed |
| identity disagreement without waveform swap | verifier says identity drift, waveform still favors target |
| content-only unconfirmed by waveform | SSL content says mismatch, waveform does not agree |
| single-verifier / speaker-verifier instability | ECAPA-like local disagreement without stronger confirmation |
| non-mismatch or unconfirmed failure | low SI-SNRi without current mismatch evidence |
| no confirmed mismatch | no reliable mismatch evidence |

## Conclusion

Extreme negative samples are mostly mismatch-driven, but the mismatch type differs by condition.

```text
similar-speaker extreme failures: mostly identity mismatch
similar-content extreme failures: mostly content-attribution mismatch
relative SI-SNRi decline: useful failure signal, but not automatically mismatch
```

This supports a condition-aware diagnostic framework rather than one universal verifier.
