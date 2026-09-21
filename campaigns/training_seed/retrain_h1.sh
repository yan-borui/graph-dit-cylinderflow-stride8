#!/usr/bin/env bash
# Run exactly one explicitly requested stage of the locked personal workflow.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
: "${DATA_DIR:?Set DATA_DIR to the existing stride8 Train/Validation dataset}"
: "${AUTOENCODER:?Set AUTOENCODER explicitly to the newly exported dit_autoencoder.pt}"
: "${ARTIFACTS_DIR:?Set a new ARTIFACTS_DIR for this representation}"
: "${RUN_DIR:?Set a new personal RUN_DIR}"
if [[ ! -f "$AUTOENCODER" ]]; then
    printf '%s\n' "Missing new VGAE export: $AUTOENCODER" >&2
    exit 2
fi
action=${1:?Usage: bash scripts/retrain_h1.sh prepare|train|resume}
if (( $# != 1 )); then
    printf '%s\n' 'This entry fixes the configuration and the 1M endpoint.' >&2
    exit 2
fi
python_bin=${PYTHON:-python}
config_file=${CONFIG:-"$code_root/configs/h1_w512_d24_cosine1m_uvp_c4.json"}
case "$action" in
    prepare|train|resume) ;;
    *) printf '%s\n' "Unknown stage: $action" >&2; exit 2 ;;
esac
mkdir -p "${RUN_DIR}.launcher_logs"
attempt_dir=$(mktemp -d "${RUN_DIR}.launcher_logs/${action}_XXXXXX")
exec > >(tee "$attempt_dir/launcher.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$attempt_dir/exit_code.txt"' EXIT
if [[ "$action" == prepare ]]; then
    "$python_bin" -m graph_dit.representation prepare \
        --data-dir "$DATA_DIR" --autoencoder "$AUTOENCODER" \
        --output-dir "$ARTIFACTS_DIR" --config "$config_file" --device cuda:0
else
    "$python_bin" -m graph_dit.representation verify \
        --data-dir "$DATA_DIR" --autoencoder "$AUTOENCODER" \
        --artifacts "$ARTIFACTS_DIR" --config "$config_file"
    extra=()
    if [[ "$action" == resume ]]; then extra=(--resume); fi
    "$python_bin" -m graph_dit.train --config "$config_file" \
        --artifacts "$ARTIFACTS_DIR" --data-dir "$DATA_DIR" \
        --output-dir "$RUN_DIR" --stage-end-updates 1000000 --device cuda:0 "${extra[@]}"
fi
