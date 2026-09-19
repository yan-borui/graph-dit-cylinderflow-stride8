#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8%1
set -euo pipefail
if (( $# != 1 )); then
    printf 'Usage: sbatch scripts/slurm_ablation_4gpu.sh preflight|train|resume|evaluate\n' >&2
    exit 2
fi
action=$1
case "$action" in
    preflight|train|resume|evaluate) ;;
    *) printf 'Unsupported Slurm action: %s\n' "$action" >&2; exit 2 ;;
esac
tasks=(h1_seed0 h2_seed0 full_seed0 h1_seed1 h2_seed1 full_seed1 h1_seed2 h2_seed2 full_seed2)
index=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}
if (( index < 0 || index > 8 )); then printf 'Array index must be 0..8\n' >&2; exit 2; fi
if [[ "$action" == preflight ]] && (( index > 2 )); then
    printf 'Capacity/restore acceptance is shared by seeds of the same architecture; skipping index %s.\n' "$index"
    exit 0
fi
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$code_root/scripts/ablation_4gpu.sh" "$action" "${tasks[$index]}"
