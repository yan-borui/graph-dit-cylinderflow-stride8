#!/usr/bin/env bash
# Verify a frozen representation, accept the production path, and train a new seed.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root"
: "${PYTHON:?Set the production interpreter}"
: "${CONFIG:?Set the selected training-seed configuration}"
: "${RUN_DIR:?Set a fresh training result directory}"
: "${EXPERIMENT_DIR:?Set the independent experiment root}"
: "${LAUNCH_DIR:?Set the launcher evidence directory}"
: "${ACCEPTANCE_DIR:?Set a fresh acceptance directory}"
mkdir -p "$LAUNCH_DIR"
if [[ -e "$LAUNCH_DIR/launch.pid" ]]; then
    printf '%s\n' 'This launcher has already been used; inspect its stage and exit first.' >&2
    exit 3
fi
printf '%s\n' "$$" > "$LAUNCH_DIR/launch.pid"
exec > >(tee -a "$LAUNCH_DIR/launch.log") 2>&1
stage=environment
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$LAUNCH_DIR/launch.exit"
    printf '%s\t%s\t%s\n' "$(date -Is)" "$stage" "$rc" >> "$LAUNCH_DIR/stages.tsv"
}
trap finish EXIT
date -Is
nvidia-smi > "$LAUNCH_DIR/nvidia-smi.txt"
"$PYTHON" -m pip freeze > "$LAUNCH_DIR/packages.txt"
"$PYTHON" -c 'import json,platform,sys,torch; print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count()), indent=2))' > "$LAUNCH_DIR/environment.json"
cp "$CONFIG" "$LAUNCH_DIR/config.json"
run_stage() {
    stage=$1
    shift
    printf '%s\n' "$stage" > "$LAUNCH_DIR/current_stage"
    printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$LAUNCH_DIR/stages.tsv"
    "$@" &
    child=$!
    printf '%s\n' "$child" > "$LAUNCH_DIR/$stage.pid"
    if wait "$child"; then rc=0; else rc=$?; fi
    printf '%s\n' "$rc" > "$LAUNCH_DIR/$stage.exit"
    printf '%s\t%s\t%s\n' "$(date -Is)" "$stage" "$rc" >> "$LAUNCH_DIR/stages.tsv"
    if (( rc != 0 )); then exit "$rc"; fi
}
run_stage verify "$PYTHON" -m graph_dit.representation verify \
    --data-dir "$DATA_DIR" --autoencoder "$AUTOENCODER" \
    --artifacts "$ARTIFACTS_DIR" --config "$CONFIG"
run_stage acceptance "$PYTHON" -m graph_dit.preflight \
    --config "$CONFIG" \
    --artifacts "$ARTIFACTS_DIR" --data-dir "$DATA_DIR" \
    --output-dir "$ACCEPTANCE_DIR" --device cuda:0 --updates 8
run_stage train bash scripts/retrain_h1.sh train
stage=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$RUN_DIR/status.json")
printf '%s\n' "$stage" > "$LAUNCH_DIR/current_stage"
