#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/graph_latent_diffusion_4gpu.sh <airfoil|cylinderflow> <prepare|train|resume|eval|benchmark>" >&2
}
if [[ $# != 2 ]]; then usage; exit 2; fi
dataset=$1
action=$2
case "$dataset" in airfoil|cylinderflow) ;; *) usage; exit 2 ;; esac
case "$action" in prepare|train|resume|eval|benchmark) ;; *) usage; exit 2 ;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python=${PYTHON:-python}
work=${WORK_DIR:-"$repo/runs/graph_latent_diffusion"}
export DATA_DIR=${DATA_DIR:-"$work/$dataset/data"}
export CACHE_DIR=${CACHE_DIR:-"$work/$dataset/train-cache"}
run=${RESULT_ROOT:-"$work/$dataset/graph-diffusion-seed0"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

case "$action" in
  prepare)
    exec bash scripts/deterministic_4gpu.sh "$dataset" prepare
    ;;
  train|resume)
    "$python" -c 'import sys, torch; n = torch.cuda.device_count(); sys.exit(0 if n >= 4 else f"four visible CUDA devices required for training; found {n}")'
    bash scripts/deterministic_4gpu.sh "$dataset" prepare
    options=()
    if [[ "$action" == resume ]]; then options+=(--resume); fi
    if [[ -n "${VERIFICATION_UPDATES:-}" ]]; then
      options+=(--verification-updates "$VERIFICATION_UPDATES")
    fi
    exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
      --module graph_latent_diffusion.train --dataset "$dataset" --data-dir "$DATA_DIR" \
      --cache-dir "$CACHE_DIR" --result-root "$run" "${options[@]}"
    ;;
  eval|benchmark)
    steps=${SAMPLING_STEPS:-6}
    members=${ENSEMBLE_SIZE:-8}
    split=${SPLIT:-validation}
    data=$DATA_DIR
    if [[ "$split" == test ]]; then data=${TEST_DATA_DIR:-$data}; fi
    output=${OUTPUT_DIR:-"$run/reports/${action}_${split}_s${steps}_k${members}"}
    exec "$python" -m graph_latent_diffusion.evaluation "$action" --dataset "$dataset" \
      --data-dir "$data" --cache-dir "$CACHE_DIR" --result-root "$run" --split "$split" \
      --sampling-steps "$steps" --ensemble-size "$members" --output-dir "$output"
    ;;
esac
