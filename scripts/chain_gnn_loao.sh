#!/usr/bin/env bash
# Chain 2: wait for the GNN LOCO CV to finish, then run GNN LOAO CV with the
# same locked config.
set -u

echo "[chain] waiting for runs/summary_loco.json ..."
until [[ -f runs/summary_loco.json ]]; do sleep 60; done
echo "[chain] GNN LOCO done, starting LOAO $(date)"

EXTRA="--node-dropout 0 --edge-dropout 0" bash scripts/run_cv.sh loao \
    || { echo "[chain] GNN LOAO FAILED"; exit 1; }
echo "[chain] GNN LOAO done $(date)"
