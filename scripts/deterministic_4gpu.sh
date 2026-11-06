#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/deterministic_4gpu.sh <airfoil|cylinderflow> <prepare|train|resume|eval|benchmark>" >&2
}
if [[ $# != 2 ]]; then usage; exit 2; fi
dataset=$1
action=$2
case "$dataset" in airfoil|cylinderflow) ;; *) usage; exit 2 ;; esac
case "$action" in prepare|train|resume|eval|benchmark) ;; *) usage; exit 2 ;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python=${PYTHON:-python}
work=${WORK_DIR:-"$repo/runs/deterministic_joint"}
data=${DATA_DIR:-"$work/$dataset/data"}
cache=${CACHE_DIR:-"$work/$dataset/train-cache"}
run=${RESULT_ROOT:-"$work/$dataset/transformer_seed0"}
if [[ "$dataset" == airfoil ]]; then
  : "${PAPER_REPRESENTATION_ID:?Set the Airfoil representation_id from the selected paper run.}"
  : "${PAPER_ARTIFACT_ID:?Set the Airfoil artifact_id from the same selected paper run.}"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

prepare_cache() {
  : "${PAPER_ARTIFACTS:?Set PAPER_ARTIFACTS to the selected paper run directory containing artifact.json and its frozen VGAE export.}"
  "$python" -m deterministic_joint.data --dataset "$dataset" --data-dir "$data" \
    --paper-artifacts "$PAPER_ARTIFACTS" --cache-dir "$cache" --device cuda:0
}

case "$action" in
  prepare)
    prepare_cache
    ;;
  train|resume)
    "$python" -c 'import sys, torch; n = torch.cuda.device_count(); sys.exit(0 if n >= 4 else f"four visible CUDA devices required for training; found {n}")'
    prepare_cache
    options=()
    if [[ "$action" == resume ]]; then options+=(--resume); fi
    if [[ -n "${VERIFICATION_UPDATES:-}" ]]; then
      options+=(--verification-updates "$VERIFICATION_UPDATES")
    fi
    exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
      --module deterministic_joint.train --dataset "$dataset" --data-dir "$data" \
      --cache-dir "$cache" --result-root "$run" "${options[@]}"
    ;;
  eval|benchmark)
    split=${SPLIT:-validation}
    if [[ "$split" == test ]]; then data=${TEST_DATA_DIR:-$data}; fi
    output=${OUTPUT_DIR:-"$run/reports/${action}_$split"}
    exec "$python" -m deterministic_joint.evaluation "$action" --dataset "$dataset" \
      --data-dir "$data" --cache-dir "$cache" --result-root "$run" --split "$split" \
      --output-dir "$output"
    ;;
esac
