#!/usr/bin/env bash
# Use only after confirming the assigned four-GPU production environment.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
: "${VERIFICATION_ROOT:?Set a new directory, separate from every formal run}"
: "${AUTOENCODER:?Set the selected Ellipse VGAE export}"
: "${ARTIFACTS_DIR:?Set its matching prepared latent cache}"
: "${DATA_DIR:?Set the full prepared Ellipse dataset}"
variant=${VARIANT:-deterministic}
bash scripts/ellipse_experiments.sh train --variant "$variant" --seed 0 \
    --cohort "$VERIFICATION_ROOT" --stage-end-updates 1
bash scripts/ellipse_experiments.sh resume --variant "$variant" --seed 0 \
    --cohort "$VERIFICATION_ROOT" --stage-end-updates 2
