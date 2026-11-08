#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    data|prepare|train|resume|validate|test) ;;
    *) printf 'Usage: bash scripts/ellipse_4gpu.sh data|prepare|train|resume|validate|test\n' >&2; exit 2 ;;
esac
export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/megaflow2d}
export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_ellipse_stride15}
if [[ "$action" != data && "$action" != prepare ]]; then
    : "${RESULT_ROOT:?Set RESULT_ROOT for this training run}"
    : "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
    if [[ "$action" == resume && ! -d "$RESULT_ROOT" ]]; then
        printf 'Resume requires the existing RESULT_ROOT.\n' >&2; exit 2
    fi
    if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
        printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
    fi
fi
data_options=()
if [[ "${ELLIPSE_OFFLINE:-0}" == 1 ]]; then data_options=(--offline); fi
"$python_bin" -m megaflow_data.ensure --raw-dir "$RAW_DATA_DIR" --output-dir "$DATA_DIR" "${data_options[@]}"
if [[ "$action" == data ]]; then exit 0; fi
data="$DATA_DIR/ellipse_stride15_65frames.h5"
manifest="$DATA_DIR/ellipse_stride15_65frames_manifest.json"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi
split=validation
if [[ "$action" == test ]]; then split=test; fi
if [[ "$action" == validate || "$action" == test ]]; then
    export EVAL_DIR=${EVAL_DIR:-$RESULT_ROOT/${split}_ellipse_4gpu}
fi


if [[ -z "${AUTOENCODER:-}" ]]; then
    : "${VGAE_REPO:?Set VGAE_REPO for automatic VGAE-to-DiT training, or AUTOENCODER for an existing export}"
    : "${VGAE_RESULT_ROOT:?Set a separate VGAE_RESULT_ROOT}"
    export AUTOENCODER="$VGAE_RESULT_ROOT/campaign/runs/w512_d4-4-2_c4_seed0/dit_autoencoder.pt"
    if [[ ! -f "$AUTOENCODER" ]]; then
        vgae_action=train
        if [[ -d "$VGAE_RESULT_ROOT" ]]; then vgae_action=resume; fi
        if [[ "$action" == prepare || "$action" == validate || "$action" == test ]]; then
            printf 'Prepare/evaluation requires an existing VGAE export.\n' >&2; exit 2
        fi
        RESULT_ROOT="$VGAE_RESULT_ROOT" bash "$VGAE_REPO/scripts/ellipse_4gpu.sh" "$vgae_action"
    fi
fi
export ARTIFACTS_DIR=${ARTIFACTS_DIR:-${AUTOENCODER}.ellipse_latents}
config=${ELLIPSE_CONFIG:-$code_root/configs/ellipse_h1_w512_d24_4gpu.json}
mkdir -p "$(dirname -- "$ARTIFACTS_DIR")"
"$python_bin" "$code_root/airfoil_data/portable_lock.py" --wait --lock "${ARTIFACTS_DIR}.lock" -- \
    bash "$code_root/scripts/prepare_ellipse_cache.sh"
"$python_bin" -m graph_dit.representation verify --data-dir "$DATA_DIR" \
    --autoencoder "$AUTOENCODER" --artifacts "$ARTIFACTS_DIR" --config "$config"
if [[ "$action" == prepare ]]; then exit 0; fi
if [[ "$action" == validate || "$action" == test ]]; then
    exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=4 \
        --module graph_dit.evaluate_ellipse --run "$RESULT_ROOT" --data-dir "$DATA_DIR" \
        --artifacts "$ARTIFACTS_DIR" --split "$split" --output-dir "$EVAL_DIR"
fi
mkdir -p "$(dirname -- "$RESULT_ROOT")"
log_dir=$(mktemp -d "${RESULT_ROOT}_launcher_XXXXXX")
set +e
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --log-dir "$log_dir" --tee 3 --module graph_dit.train --config "$config" \
    --artifacts "$ARTIFACTS_DIR" --data-dir "$DATA_DIR" --output-dir "$RESULT_ROOT" \
    --stage-end-updates 250000 --device cuda "${extra[@]}" 2>&1 | tee "$log_dir/launcher.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$log_dir/exit_code.txt"
exit "$rc"
