#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
python_bin=${PYTHON:-python}
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"
config=${AIRFOIL_CONFIG:-$code_root/configs/airfoil_h1_w512_d24_4gpu.json}
if [[ ! -f "$ARTIFACTS_DIR/train_latents.h5" ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Set allocated GPU IDs for representation preparation}"
    if [[ -e "$ARTIFACTS_DIR" ]]; then
        printf 'Incomplete representation exists: %s; choose a new ARTIFACTS_DIR.\n' "$ARTIFACTS_DIR" >&2; exit 2
    fi
    attempt=$(mktemp -d "${ARTIFACTS_DIR}.attempt.XXXXXX")
    "$python_bin" -m graph_dit.representation prepare --data-dir "$DATA_DIR" \
        --autoencoder "$AUTOENCODER" --output-dir "$attempt/artifacts" --device cuda:0 --config "$config" \
        2>&1 | tee "$attempt/prepare.log"
    mv -- "$attempt/artifacts" "$ARTIFACTS_DIR"
fi
