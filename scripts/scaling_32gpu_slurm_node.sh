#!/usr/bin/env bash
set -euo pipefail
export NODE_RANK=${SLURM_NODEID:?one torchrun agent per allocated node}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/scripts/scaling_32gpu_node.sh" "$@"
