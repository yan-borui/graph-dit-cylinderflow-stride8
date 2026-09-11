#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
set -euo pipefail
if [[ "$#" -ne 4 ]]; then
    echo "usage: sbatch slurm_screen_4gpu.sh PYTHON PLAN_JSON DATA_DIR ARTIFACTS" >&2
    exit 2
fi
python_bin=$1
plan_file=$2
data_dir=$3
artifacts=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$python_bin" -m graph_dit.campaign worker --plan "$plan_file" \
    --index "${SLURM_ARRAY_TASK_ID:?}" --data-dir "$data_dir" --artifacts "$artifacts" --resume-incomplete

