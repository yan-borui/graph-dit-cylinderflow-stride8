#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
action=${1:-train}
variant=${2:-h1}
if (( $# > 2 )); then
    printf 'Usage: bash scripts/airfoil_ablation_4gpu.sh prepare|train|resume|report [h1|h2|full]\n' >&2
    exit 2
fi
case "$action" in prepare|train|resume|report) ;; *) exit 2 ;; esac
case "$variant" in h1|h2|full) ;; *) printf 'Choose h1, h2, or full.\n' >&2; exit 2 ;; esac
: "${COHORT:?Set COHORT to the parent directory for the three attention runs}"
if [[ "$action" == report ]]; then
    exec "${PYTHON:-python}" "$code_root/scripts/airfoil_ablation_report.py" --cohort "$COHORT"
fi
export AIRFOIL_CONFIG="$code_root/configs/airfoil_ablation_${variant}_4gpu.json"
export RESULT_ROOT="$COHORT/${variant}_seed0"
mkdir -p "$COHORT"
# Hold one lock for this four-GPU cohort through preparation and training.
exec "${PYTHON:-python}" "$code_root/airfoil_data/portable_lock.py" \
    --lock "$COHORT/launcher.lock" -- \
    bash "$code_root/scripts/airfoil_4gpu.sh" "$action"
