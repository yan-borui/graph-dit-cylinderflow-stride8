#!/usr/bin/env bash
# Run once on each allocated node. Rank count always stays at 32.
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'usage: scaling_32gpu_node.sh train|resume|acceptance|evaluate PYTHON TASK_INDEX DATA_DIR ARTIFACTS COHORT_DIR' >&2
    exit 2
fi
action=$1
python_bin=$2
task_index=$3
data_dir=$4
artifacts=$5
cohort_dir=$6
case "$action" in train|resume|acceptance|evaluate) ;; *) exit 2 ;; esac
[[ "$task_index" =~ ^([0-9]|1[01])$ ]] || { echo 'TASK_INDEX must be 0..11' >&2; exit 2; }
: "${NNODES:?set NNODES for this allocation}"
: "${GPUS_PER_NODE:?set GPUS_PER_NODE for this allocation}"
: "${NODE_RANK:?set NODE_RANK on each node}"
: "${MASTER_ADDR:?set the same reachable rank-zero hostname on every node}"
: "${MASTER_PORT:?set a free common rendezvous port}"
for number in "$NNODES" "$GPUS_PER_NODE" "$NODE_RANK" "$MASTER_PORT"; do
    [[ "$number" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'topology fields must be decimal integers' >&2; exit 2; }
done
(( NNODES > 0 && GPUS_PER_NODE > 0 && NNODES * GPUS_PER_NODE == 32 )) || {
    echo 'NNODES * GPUS_PER_NODE must equal 32' >&2; exit 2;
}
(( NODE_RANK < NNODES && MASTER_PORT > 0 && MASTER_PORT < 65536 )) || exit 2
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=$(command -v -- "$python_bin")
data_dir=$(realpath -e -- "$data_dir")
artifacts=$(realpath -e -- "$artifacts")
mkdir -p -- "$cohort_dir"
cohort_dir=$(realpath -e -- "$cohort_dir")
models=(w512_d24 w768_d24 w1024_d24 w1024_d32)
model_id=${models[$((task_index / 3))]}
seed=$((task_index % 3))
task_id=${model_id}_seed${seed}
config=$repo_root/configs/scaling_32gpu/${task_id}.json
run_dir=$cohort_dir/runs/$task_id
record_parent=$cohort_dir/launcher/$task_id/$action/node_$NODE_RANK
mkdir -p -- "$record_parent"
record_dir=$(mktemp -d "$record_parent/attempt_XXXXXX")
exec > >(tee -a "$record_dir/launch.log") 2>&1
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$record_dir/exit_code"
}
trap finish EXIT
printf '%s\n' "$$" > "$record_dir/launcher.pid"
printf '%s\n' "action=$action" "task=$task_id" "node_rank=$NODE_RANK" \
    "nnodes=$NNODES" "gpus_per_node=$GPUS_PER_NODE" "master_addr=$MASTER_ADDR" \
    "master_port=$MASTER_PORT" "hostname=$(hostname)" "started=$(date -Is)" \
    "python=$python_bin" > "$record_dir/allocation.txt"
cp -- "$config" "$record_dir/config.json"
nvidia-smi > "$record_dir/nvidia-smi.txt"
nvidia-smi topo -m > "$record_dir/gpu-topology.txt"
"$python_bin" -m pip freeze > "$record_dir/packages.txt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
case "$action" in
    acceptance)
        mkdir -p -- "$cohort_dir/acceptance"
        module=graph_dit.preflight
        arguments=(--config "$config" --artifacts "$artifacts" --data-dir "$data_dir"
            --output-dir "$cohort_dir/acceptance/$task_id" --device cuda:0 --updates 8)
        ;;
    train|resume)
        mkdir -p -- "$cohort_dir/runs"
        module=graph_dit.train
        arguments=(--config "$config" --artifacts "$artifacts" --data-dir "$data_dir"
            --output-dir "$run_dir" --device cuda:0 --stage-end-updates 125000)
        if [[ "$action" == resume ]]; then arguments+=(--resume); fi
        ;;
    evaluate)
        module=graph_dit.scaling_evaluate
        arguments=(--run "$run_dir" --artifacts "$artifacts" --data-dir "$data_dir"
            --output-dir "$cohort_dir/validation/$task_id" --selection endpoint)
        ;;
esac
"$python_bin" -m torch.distributed.run \
    --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --max-restarts 0 \
    --module "$module" "${arguments[@]}"
