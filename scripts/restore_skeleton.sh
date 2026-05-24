#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/data/tse_cue_project}
mkdir -p "$ROOT/code/wesep-real-tse/tools" "$ROOT/experiments/configs" "$ROOT/recovered_stage1_results"
cp -av code_tools/* "$ROOT/code/wesep-real-tse/tools/" 2>/dev/null || true
cp -av configs/* "$ROOT/experiments/configs/" 2>/dev/null || true
cp -av results "$ROOT/recovered_stage1_results/"
echo "Skeleton restored to $ROOT"
echo "Large datasets are not included; rebuild/download them separately."
