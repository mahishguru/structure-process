#!/usr/bin/env bash
# Queue: wait for run 4a/4b trainers, then evaluate both on dev70 test (GPU).
set -u
PIDS=(3830776 3830777)
for pid in "${PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "[queue] trainers finished at $(date)"
for run in runs/gnn_dev70_s90_run4a_aug05 runs/gnn_dev70_s90_run4b_aug05_wd1e3; do
    echo "[queue] evaluating $run"
    .venv/bin/python -u scripts/evaluate.py --run "$run" --pipeline gnn \
        --split dev70 --scales 90 --device cuda
done
echo "[queue] done at $(date)"
for run in runs/gnn_dev70_s90_run4a_aug05 runs/gnn_dev70_s90_run4b_aug05_wd1e3; do
    echo "=== $run ==="
    cat "$run/eval_dev70.json"
done
