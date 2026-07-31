#!/usr/bin/env bash
# Sequential 5-fold LOCO (+ optional LOAO) train + eval on GPU for the GNN
# pipeline. Resume-safe: skips a fold if its eval JSON already exists.
#
# Usage:
#   scripts/run_cv.sh loco                 # 5 LOCO folds
#   scripts/run_cv.sh loao                 # all LOAO splits
#   EXTRA="--weight-decay 1e-3" scripts/run_cv.sh loco   # extra train flags
set -u
MODE="${1:-loco}"
PY=.venv/bin/python
EXTRA="${EXTRA:-}"
COMMON="--pipeline gnn --scales 90 --epochs 400 --samples-per-condition 16 --device cuda"

if [[ "$MODE" == "loco" ]]; then
    SPLITS=(loco_fold0 loco_fold1 loco_fold2 loco_fold3 loco_fold4)
else
    SPLITS=($(ls data/splits/loao_*.json | xargs -n1 basename | sed 's/.json//'))
fi

for split in "${SPLITS[@]}"; do
    out="runs/gnn_${split}_s90"
    if [[ -f "$out/eval_${split}.json" ]]; then
        echo "[cv] $split already evaluated, skipping"
        continue
    fi
    echo "[cv] === $split : training ==="
    $PY -u scripts/train.py --split "$split" $COMMON $EXTRA --out "$out" \
        || { echo "[cv] TRAIN FAILED for $split"; exit 1; }
    echo "[cv] === $split : evaluating ==="
    $PY -u scripts/evaluate.py --run "$out" --split "$split" \
        --pipeline gnn --scales 90 --device cuda \
        || { echo "[cv] EVAL FAILED for $split"; exit 1; }
done

echo "[cv] all ${MODE} folds done"
RUNS=(); for s in "${SPLITS[@]}"; do RUNS+=("runs/gnn_${s}_s90"); done
if [[ "$MODE" == "loco" ]]; then
    OUT=runs/summary_loco.json
else
    OUT=runs/summary_loao_gnn.json
fi
$PY scripts/aggregate_folds.py \
    --runs "${RUNS[@]}" --splits "${SPLITS[@]}" \
    --name gnn --out "$OUT"
