#!/usr/bin/env bash
#SBATCH --job-name=graph-dit-h1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%A_%a.log
set -Eeuo pipefail
: "${PLAN:?Set PLAN to the shared plan.json}"
: "${DATA_DIR:?Set DATA_DIR to the shared released data directory}"
: "${ARTIFACTS:?Set ARTIFACTS to the shared frozen representation directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm job array}"
REPOSITORY_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPOSITORY_ROOT}"
[[ -f graph_dit/campaign.py ]] || { echo 'Submit from the repository root, or set REPO_ROOT.' >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
args=( -m graph_dit.campaign worker --plan "${PLAN}" --index "${SLURM_ARRAY_TASK_ID}"
       --data-dir "${DATA_DIR}" --artifacts "${ARTIFACTS}" --device cuda:0 )
if [[ "${RESUME_INCOMPLETE:-0}" == "1" ]]; then
    args+=( --resume-incomplete )
fi
# Slurm owns CUDA_VISIBLE_DEVICES. Never overwrite its GPU allocation.
srun --ntasks=1 "${PYTHON_BIN:-python}" "${args[@]}"
