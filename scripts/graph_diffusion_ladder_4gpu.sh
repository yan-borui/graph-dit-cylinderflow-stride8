#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "Usage: bash scripts/graph_diffusion_ladder_4gpu.sh <cylinderflow|airfoil> <256|512|1024> <train|resume|evaluate|run|resume-run|status>" >&2
  exit 2
fi
dataset=$1
width=$2
action=$3
case "$dataset" in cylinderflow|airfoil) ;; *) exit 2 ;; esac
case "$width" in 256|512|1024) ;; *) exit 2 ;; esac
case "$action" in train|resume|evaluate|run|resume-run|status) ;; *) exit 2 ;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python=${PYTHON:-python}
export WORK_DIR=${WORK_DIR:-"$repo/runs/graph_diffusion_ladder"}
export MODEL_WIDTH=$width
export RESULT_ROOT=${RESULT_ROOT:-"$WORK_DIR/$dataset/w${width}-seed0"}
if [[ "$action" == status ]]; then
  exec "$python" -m graph_latent_diffusion.ladder status --run "$RESULT_ROOT"
fi
: "${DATA_DIR:?Set DATA_DIR to the original prepared Train/Validation65 data.}"
: "${CACHE_DIR:?Set CACHE_DIR to the original matching frozen-representation cache.}"
export DATA_DIR CACHE_DIR
split=${SPLIT:-both}
case "$split" in validation|test|both) ;; *) echo "Invalid SPLIT: $split" >&2; exit 2 ;; esac
case "$action" in
  run|resume-run|evaluate)
    if [[ "$split" != validation ]]; then
      : "${TEST_DATA_DIR:?Set TEST_DATA_DIR to the official prepared Test65 data.}"
      if [[ ! -d "$TEST_DATA_DIR" ]]; then echo "Missing TEST_DATA_DIR: $TEST_DATA_DIR" >&2; exit 2; fi
    fi
    ;;
esac
case "$action" in
  train|resume) exec bash scripts/graph_latent_diffusion_4gpu.sh "$dataset" "$action" ;;
  run) bash scripts/graph_latent_diffusion_4gpu.sh "$dataset" train ;;
  resume-run) bash scripts/graph_latent_diffusion_4gpu.sh "$dataset" resume ;;
esac
options=()
if [[ "$split" != validation ]]; then options+=(--test-data-dir "$TEST_DATA_DIR"); fi
exec "$python" -m graph_latent_diffusion.ladder evaluate \
  --dataset "$dataset" --width "$width" --run "$RESULT_ROOT" --split "$split" \
  --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --output-dir "${EVAL_OUTPUT_DIR:-$RESULT_ROOT/reports/matched_250k_s6_k1_r3}" "${options[@]}"
