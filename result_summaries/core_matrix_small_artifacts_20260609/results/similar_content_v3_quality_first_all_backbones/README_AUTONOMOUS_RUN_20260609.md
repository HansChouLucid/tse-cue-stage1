# Similar-content v3 quality-first autonomous run (2026-06-09)

Main controller:
- scripts/run_after_tts_full_pipeline.sh
- pid: full_pipeline.pid

Watchdog:
- scripts/watch_scv3_pipeline.py
- pid: watchdog.pid
- checks every 10 minutes for up to 8 hours and restarts the main controller if it exits before DONE.

Pipeline stages:
1. Wait for both F5-TTS shards to reach 2815 rows.
2. Build audio sanity CSV.
3. Run full faster-whisper-large-v3 ASR quality gate.
4. Build final quality-first v3 bundle.
5. Run full inference for pretrained USEF-TFGridNet, pretrained TFMap+Context, and BSRNN USEF/TFMap/Context.
6. Run content local diagnostics with Wav2Vec2, WavLM, HuBERT, then merge multi-verifier results.
7. Write scv3_quality_first_summary.csv and DONE.

Important guards:
- old v2 / old v3 paths are not used.
- final target/ref/interferer are aligned to Libri2Mix s1/s2 by target speaker side.
- BSRNN uses per-sample cue keys `sample_key::spk`; formal summaries keep only the target-speaker output.
