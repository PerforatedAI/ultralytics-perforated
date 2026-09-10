#!/usr/bin/env bash
################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Queue the YOLO26m and YOLO26l Cityscapes arms, dendrite then plain.          #
################################################################################
#
# Four jobs back to back on one GPU, in this order:
#   1. medium_l0_l1_l2_first   yolo26m, pos0 dendrites
#   2. large_l0_l1_l2_first    yolo26l, pos0 dendrites
#   3. medium_plain            yolo26m, no dendrites
#   4. large_plain             yolo26l, no dendrites
#
# Each size derives its config from cfg/yolo26n_cityscapes_pai.yaml the same
# way the small pair did: model swapped, batch set, and nbs matched to batch
# so accumulate stays 1. Everything else, including the plateau schedule and
# the pos0 placement, is inherited.
#
# Batch sizes are measured, not halved from small. On 2026-09-07 medium at
# batch 24 raised CUDA OOM and ultralytics silently retried at 12 with nbs
# still 24 (accumulate 2, args.yaml still saying 24), so that launch was
# killed. Medium at 12 reserved 13.5 GB of the A10G's 23 GB. Large was probed
# through train_pai.py with the real config: batch 12 reserved 16.5 GB, no OOM.
#
# Launch inside the meow tmux session, where the PerforatedAI license
# environment variables live. A fresh shell blocks on the license prompt.
#   tmux send-keys -t meow 'bash tools/run_ml_arms.sh > runs/queue/ml_driver.log 2>&1' Enter
#
# Delete this script once the four runs are verified. The derived configs
# in the queue dir and each run's args.yaml are the record.

set -u -o pipefail

python_bin=.venv/bin/python
config_base=cfg/yolo26n_cityscapes_pai.yaml
LARGE_BATCH=12
log_dir=runs/queue/$(date +%Y%m%d_%H%M%S)_ml
mkdir -p "$log_dir"
echo "Medium/large arms logging to $log_dir"

# derive <size> <batch>: writes $log_dir/yolo26<size>_cityscapes_pai.yaml
derive() {
    local size=$1 batch=$2
    local cfg=$log_dir/yolo26${size}_cityscapes_pai.yaml
    sed -e "s/^model: .*/model: yolo26${size}.pt/" \
        -e "s/^batch: .*/batch: ${batch}/" \
        "$config_base" > "$cfg"
    printf '\n# nbs matched to batch so accumulate is round(%s / %s) = 1\nnbs: %s\n' \
        "$batch" "$batch" "$batch" >> "$cfg"
    echo "Config $cfg differs from $config_base by:"
    diff "$config_base" "$cfg"
}
derive m 12
derive l "$LARGE_BATCH"

# run <job-index> <run-name> <config> [extra train_pai.py args...]
run() {
    local idx=$1 name=$2 cfg=$3
    shift 3
    local log=$log_dir/${idx}_${name}.log
    echo "[$(date +%H:%M:%S)] job $idx/4 $name -> $log"
    echo "  $python_bin train_pai.py $cfg --name $name $*"
    $python_bin train_pai.py "$cfg" --name "$name" "$@" > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] $name exit $?"
}

pai_args=(--perforate-model --n-epochs-to-switch 15 --initial-correlation-batches 20)
run 1 medium_l0_l1_l2_first "$log_dir/yolo26m_cityscapes_pai.yaml" "${pai_args[@]}"
run 2 large_l0_l1_l2_first  "$log_dir/yolo26l_cityscapes_pai.yaml" "${pai_args[@]}"
run 3 medium_plain          "$log_dir/yolo26m_cityscapes_pai.yaml"
run 4 large_plain           "$log_dir/yolo26l_cityscapes_pai.yaml"

echo
echo "================================ RESULTS ================================"
for r in medium_l0_l1_l2_first large_l0_l1_l2_first medium_plain large_plain; do
    csv=runs/detect/yolo26-cityscapes/$r/results.csv
    if [ -f "$csv" ]; then
        echo "$r: $(($(wc -l < "$csv") - 1)) epochs, best mAP50-95 / mAP75"
        head -1 "$csv" | cut -d, -f1,6,7,8,9,10
        sort -t, -k9 -g "$csv" | tail -1 | cut -d, -f1,6,7,8,9,10
    else
        echo "$r: no results.csv"
    fi
done
echo "Medium/large arms finished at $(date)"
