#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
: "${DATA_DIR:?Set DATA_DIR to the prepared Airfoil directory}"
python_bin=${PYTHON:-python}
action=${1:-train}
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"

: "${ARTIFACTS_DIR:?Set ARTIFACTS_DIR for the Airfoil Train latent cache}"
: "${AUTOENCODER:?Set AUTOENCODER to the Airfoil VGAE dit_autoencoder.pt export}"
config="$code_root/configs/airfoil_h1_w512_d24_4gpu.json"
case "$action" in
    prepare)
        exec "$python_bin" -m graph_dit.representation prepare --data-dir "$DATA_DIR" --autoencoder "$AUTOENCODER" --output-dir "$ARTIFACTS_DIR" --device cuda:0 --config "$config"
        ;;
    train|resume) ;;
    *) printf 'Usage: bash scripts/airfoil_4gpu.sh prepare|train|resume\n' >&2; exit 2 ;;
esac

: "${RESULT_ROOT:?Set a new RESULT_ROOT for this training run}"
: "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi

"$python_bin" -m graph_dit.representation verify --data-dir "$DATA_DIR" --autoencoder "$AUTOENCODER" --artifacts "$ARTIFACTS_DIR" --config "$config"
mkdir -p "$(dirname -- "$RESULT_ROOT")"
log_dir=$(mktemp -d "${RESULT_ROOT}_launcher_XXXXXX")
set +e
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --log-dir "$log_dir" --tee 3 --module graph_dit.train --config "$config" --artifacts "$ARTIFACTS_DIR" --data-dir "$DATA_DIR" --output-dir "$RESULT_ROOT" --stage-end-updates 250000 --device cuda "${extra[@]}" 2>&1 | tee "$log_dir/launcher.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$log_dir/exit_code.txt"
exit "$rc"
