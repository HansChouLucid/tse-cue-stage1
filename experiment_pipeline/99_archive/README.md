# Archive

This folder stores old drafts, early pilots, and documents that have been replaced by the current pipeline.

Use the current pipeline first:

```text
README.md
00_overview/
03_pretrained_usef_diagnostic/
05_reports_and_figures/condition_aware_failure_taxonomy/
```

## initial_long_notes

Early long notes and draft plans:

```text
initial mismatch definitions
metric script drafts
pilot experiment notes
remote result index
```

These ideas have been condensed into the current overview and USEF-only diagnostic pages.

## superseded_2026-05-21_old_pilots

Old model-output mismatch pilot notes. These mainly validated smoke runs and weak baseline metric plumbing.

They should not be used as current evidence because the weak baseline was trained only briefly and is not a fair comparison against pretrained USEF.

## verifier sanity note

Chunk-level verifier sanity is no longer archived. It has been restored to the main pipeline as:

```text
02_pilot_verifier_sanity/
```

Reason:

```text
verifier stability at chunk level is a core assumption of local mismatch diagnosis.
```
