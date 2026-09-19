#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=${PYTHON:-python}
action=${1:-train}
task=${2:-all}
if (( $# > 2 )); then
    printf 'Usage: bash scripts/ablation_4gpu.sh prepare|preflight|train|resume|evaluate|report [TASK|all]\n' >&2
    exit 2
fi
: "${COHORT:?Set COHORT to a new campaign result directory}"
case "$action" in
    prepare|preflight|train|resume|evaluate|report) ;;
    *) printf 'Unknown action: %s\n' "$action" >&2; exit 2 ;;
esac
if [[ "$action" == report ]]; then
    extra=()
    if [[ "${MOVIES:-0}" == 1 ]]; then extra+=(--movies); fi
    if [[ -n "${REPORT_DIR:-}" ]]; then extra+=(--output-dir "$REPORT_DIR"); fi
    exec "$python_bin" -m graph_dit.ablation report --cohort "$COHORT" "${extra[@]}"
fi
: "${DATA_DIR:?Set DATA_DIR to the shared stride8 data directory}"
: "${ARTIFACTS_DIR:?Set ARTIFACTS_DIR to the shared epoch1180 cache directory}"
if [[ "$action" != prepare ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Use the scheduler mask or four explicitly allocated GPU IDs}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then
        printf 'Exactly four allocated GPUs must be visible.\n' >&2
        exit 2
    fi
fi
common=(--cohort "$COHORT" --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS_DIR")
if [[ "$action" == prepare || "$action" == preflight || "$action" == train || "$action" == resume ]]; then
    extra=()
    if [[ -n "${AUTOENCODER:-}" ]]; then extra+=(--autoencoder "$AUTOENCODER"); fi
    mkdir -p "$COHORT/launcher"
    prepare_log=$(mktemp -d "$COHORT/launcher/prepare_XXXXXX")
    set +e
    "$python_bin" -m graph_dit.ablation prepare "${common[@]}" "${extra[@]}" 2>&1 | tee "$prepare_log/prepare.log"
    prepare_rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$prepare_rc" > "$prepare_log/exit_code.txt"
    if (( prepare_rc != 0 )); then exit "$prepare_rc"; fi
    if [[ "$action" == prepare ]]; then exit 0; fi
fi
exec "$python_bin" -m graph_dit.ablation "$action" --task "$task" "${common[@]}"
