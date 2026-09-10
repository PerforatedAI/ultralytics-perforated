#!/usr/bin/env bash
################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Queue the YOLO26n Cityscapes PerforatedAI experiment end to end.             #
################################################################################
#
# Runs three jobs back to back on one GPU:
#   1. Forced-cycle smoke, 40 epochs, switching every 6 epochs
#   2. Production dendrite arm, 300 epochs
#   3. Production plain arm, 300 epochs, matched
#
# Job 1 gates jobs 2 and 3. It is the only run that reaches the dendrite to
# neuron half of a restructure, so if it crashes or never gets there, the
# production arms would spend seven hours on the same broken path.
#
# Launch on the box, detached:
#   tmux new -d -s yolo26_queue 'bash tools/run_experiment_queue.sh'
#   tmux attach -t yolo26_queue

set -u -o pipefail

python_bin=.venv/bin/python
config_smoke=cfg/yolo26n_cityscapes_pai_smoke.yaml
config_prod=cfg/yolo26n_cityscapes_pai.yaml
log_dir=runs/queue/$(date +%Y%m%d_%H%M%S)

mkdir -p "$log_dir"
echo "Queue logging to $log_dir"

# Job 1, the gate
smoke_log=$log_dir/1_cycles_smoke.log
echo "[$(date +%H:%M:%S)] job 1/3 forced-cycle smoke -> $smoke_log"
$python_bin train_pai.py "$config_smoke" \
    --perforate-model \
    --fixed-switch-every 6 \
    --initial-correlation-batches 20 > "$smoke_log" 2>&1
smoke_status=$?

restructures=$(grep -c 'PerforatedAI restructured at epoch' "$smoke_log")
returned_to_neuron=$(grep -cE 'mode n,.*num_cycles: [1-9]' "$smoke_log")

echo "[$(date +%H:%M:%S)] smoke exit $smoke_status, $restructures restructures, $returned_to_neuron neuron epochs after a cycle"

if [ "$smoke_status" -ne 0 ] || [ "$restructures" -lt 2 ] || [ "$returned_to_neuron" -lt 1 ]; then
    echo "GATE FAILED. Production arms not launched."
    echo "Expected exit 0, at least 2 restructures, and at least 1 neuron"
    echo "epoch after num_cycles reached 1. Read $smoke_log."
    exit 1
fi
echo "GATE PASSED. The dendrite to neuron path ran. Launching production."

# Job 2, dendrite arm
pai_log=$log_dir/2_pai_arm.log
echo "[$(date +%H:%M:%S)] job 2/3 dendrite arm -> $pai_log"
$python_bin train_pai.py "$config_prod" \
    --name yolo26n_cityscapes_pai \
    --perforate-model \
    --n-epochs-to-switch 15 \
    --initial-correlation-batches 20 > "$pai_log" 2>&1
pai_status=$?
echo "[$(date +%H:%M:%S)] dendrite arm exit $pai_status"

# Job 3, matched plain arm. Runs whatever job 2 did, so a dendrite-arm
# failure still leaves the baseline in hand
plain_log=$log_dir/3_plain_arm.log
echo "[$(date +%H:%M:%S)] job 3/3 plain arm -> $plain_log"
$python_bin train_pai.py "$config_prod" \
    --name yolo26n_cityscapes_plain > "$plain_log" 2>&1
plain_status=$?
echo "[$(date +%H:%M:%S)] plain arm exit $plain_status"

echo
echo "================================ RESULTS ================================"
for run in yolo26n_cityscapes_cycles yolo26n_cityscapes_pai yolo26n_cityscapes_plain; do
    csv=runs/detect/yolo26-cityscapes/$run/results.csv
    if [ -f "$csv" ]; then
        echo "$run: $(wc -l < "$csv") rows, last epoch"
        tail -1 "$csv"
    else
        echo "$run: no results.csv"
    fi
done
echo "Queue finished at $(date)"
