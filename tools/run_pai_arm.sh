#!/usr/bin/env bash
################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Run one YOLO26n Cityscapes PerforatedAI arm, gated by a forced-cycle smoke.  #
################################################################################
#
# Runs two jobs back to back on one GPU:
#   1. Forced-cycle smoke, 8 epochs, switching every 2
#   2. The named dendrite arm on the production config
#
# Job 1 gates job 2. It is the only run short enough to reach the dendrite to
# neuron half of a restructure quickly, so if the placement in pai/setup.py
# crashes or never gets there, the production arm would spend hours on the
# same broken path.
#
# The plain baseline is already in hand at
# runs/detect/yolo26-cityscapes/yolo26n_cityscapes_plain, so no matched plain
# arm is run here. Every arm is a dendrite-placement ablation against it.
#
# Launch inside the meow tmux session, which is where the PerforatedAI license
# environment variables live. A fresh shell blocks on the license prompt.
#   tmux send-keys -t meow 'bash tools/run_pai_arm.sh <arm-name>' Enter

set -u -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <arm-name>"
    exit 2
fi
arm=$1

python_bin=.venv/bin/python
config_smoke=cfg/yolo26n_cityscapes_pai_smoke.yaml
config_prod=cfg/yolo26n_cityscapes_pai.yaml
log_dir=runs/queue/$(date +%Y%m%d_%H%M%S)_$arm

mkdir -p "$log_dir"
echo "Arm $arm logging to $log_dir"

# Job 1, the gate. Same recipe as the production arm apart from the epoch
# ceiling and the forced switch, so it exercises the real placement
gate_config=$log_dir/gate.yaml
sed -e 's/^epochs: .*/epochs: 8/' \
    -e "s/^name: .*/name: ${arm}_gate/" \
    -e 's/^plots: .*/plots: False/' \
    "$config_smoke" > "$gate_config"

gate_log=$log_dir/1_gate.log
echo "[$(date +%H:%M:%S)] job 1/2 forced-cycle gate -> $gate_log"
WANDB_MODE=disabled $python_bin train_pai.py "$gate_config" \
    --perforate-model \
    --fixed-switch-every 2 \
    --initial-correlation-batches 20 > "$gate_log" 2>&1
gate_status=$?

wrapped=$(grep -oE 'PerforatedAI wrapped [0-9]+ modules' "$gate_log" | head -1)
restructures=$(grep -c 'PerforatedAI restructured at epoch' "$gate_log")
returned_to_neuron=$(grep -cE 'mode n,.*num_cycles: [1-9]' "$gate_log")

echo "[$(date +%H:%M:%S)] gate exit $gate_status, $wrapped, $restructures restructures, $returned_to_neuron neuron epochs after a cycle"

if [ "$gate_status" -ne 0 ] || [ "$restructures" -lt 3 ] || [ "$returned_to_neuron" -lt 1 ]; then
    echo "GATE FAILED. Production arm not launched."
    echo "Expected exit 0, at least 3 restructures, and at least 1 neuron"
    echo "epoch after num_cycles reached 1. Read $gate_log."
    exit 1
fi
echo "GATE PASSED. The dendrite to neuron path ran. Launching production."

# Job 2, the arm itself
arm_log=$log_dir/2_$arm.log
echo "[$(date +%H:%M:%S)] job 2/2 dendrite arm -> $arm_log"
$python_bin train_pai.py "$config_prod" \
    --name "$arm" \
    --perforate-model \
    --n-epochs-to-switch 15 \
    --initial-correlation-batches 20 > "$arm_log" 2>&1
arm_status=$?
echo "[$(date +%H:%M:%S)] dendrite arm exit $arm_status"

echo
echo "================================ RESULTS ================================"
for run in "${arm}_gate" "$arm" yolo26n_cityscapes_plain; do
    csv=runs/detect/yolo26-cityscapes/$run/results.csv
    if [ -f "$csv" ]; then
        echo "$run: $(($(wc -l < "$csv") - 1)) epochs, best mAP50-95 / mAP75"
        head -1 "$csv" | cut -d, -f1,6,7,8,9,10
        sort -t, -k9 -g "$csv" | tail -1 | cut -d, -f1,6,7,8,9,10
    else
        echo "$run: no results.csv"
    fi
done
echo "Arm $arm finished at $(date)"
