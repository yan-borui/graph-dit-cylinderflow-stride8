#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11
#SBATCH --output=scaling-%A_%a.out
#SBATCH --error=scaling-%A_%a.err
#SBATCH --open-mode=append
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo 'usage: sbatch slurm_scaling_32gpu.sh train|resume|acceptance|evaluate PYTHON DATA_DIR ARTIFACTS COHORT_DIR' >&2
    exit 2
fi
action=$1
python_bin=$2
data_dir=$3
artifacts=$4
cohort_dir=$5
# sbatch copies this script into its spool directory; use the submission checkout.
repo_root=${SCALING_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit from the scaling checkout}}
repo_root=$(cd -- "$repo_root" && pwd)
[[ -f "$repo_root/scripts/scaling_32gpu_node.sh" ]] || exit 2
export NNODES=${SLURM_NNODES:?}
export GPUS_PER_NODE=${SCALING_GPUS_PER_NODE:-8}
(( NNODES * GPUS_PER_NODE == 32 )) || { echo 'allocation must total 32 GPUs' >&2; exit 2; }
mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
export MASTER_ADDR=${SCALING_MASTER_ADDR:-${allocated_hosts[0]}}
export MASTER_PORT=${SCALING_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# One torchrun agent per node; each agent spawns GPUS_PER_NODE DDP ranks.
# NODE_RANK is expanded inside the launched node's shell, not by the submission shell.
srun --nodes "$NNODES" --ntasks "$NNODES" --ntasks-per-node 1 \
    --gpus-per-task "$GPUS_PER_NODE" --kill-on-bad-exit=1 \
    bash "$repo_root/scripts/scaling_32gpu_slurm_node.sh" \
    "$action" "$python_bin" "${SLURM_ARRAY_TASK_ID:?}" "$data_dir" "$artifacts" "$cohort_dir"
