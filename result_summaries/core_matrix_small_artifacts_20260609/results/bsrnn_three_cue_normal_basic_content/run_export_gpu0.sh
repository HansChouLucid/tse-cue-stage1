#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/normal_basic_content_baseline_20260604
python "$BASE/export_normal_basic_bundle.py" --cue usef_ft13 --gpu 0 --base "$BASE" --manifest "$BASE/splits/basic_test_1k.jsonl"
python "$BASE/export_normal_basic_bundle.py" --cue context_ft13 --gpu 0 --base "$BASE" --manifest "$BASE/splits/basic_test_1k.jsonl"
