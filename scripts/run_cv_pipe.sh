#!/usr/bin/env bash
# Sequential CV train + eval for the conventional / genai pipelines with the
# shared flow_ar head. Resume-safe: skips a fold if its eval JSON exists.
#
# Usage: scripts/run_cv_pipe.sh <conventional|genai> <loco|loao> <lr> <wd> <spc>
set -u
PIPE="$1"; MODE="$2"; LR="$3"; WD="$4"; SPC="$5"
PY=.venv/bin/python
COMMON="--pipeline $PIPE --epochs 400 --device cuda --lr $LR --weight-decay $WD --samples-per-condition $SPC"

if [[ "$MODE" == "loco" ]]; then
    SPLITS=(loco_fold0 loco_fold1 loco_fold2 loco_fold3 loco_fold4)
else
    SPLITS=($(ls data/splits/loao_*.json | xargs -n1 basename | sed 's/.json//'))
fi

for split in "${SPLITS[@]}"; do
    out="runs/${PIPE}_${split}"
    if [[ -f "$out/eval_${split}.json" ]]; then
        echo "[cv:$PIPE] $split already evaluated, skipping"
        continue
    fi
    echo "[cv:$PIPE] === $split : training ==="
    $PY -u scripts/train.py --split "$split" $COMMON --out "$out" \
        || { echo "[cv:$PIPE] TRAIN FAILED for $split"; exit 1; }
    echo "[cv:$PIPE] === $split : evaluating ==="
    $PY -u scripts/evaluate.py --run "$out" --split "$split" \
        --pipeline "$PIPE" --device cuda \
        || { echo "[cv:$PIPE] EVAL FAILED for $split"; exit 1; }
done

echo "[cv:$PIPE] all ${MODE} folds done"
RUNS=(); for s in "${SPLITS[@]}"; do RUNS+=("runs/${PIPE}_${s}"); done
$PY scripts/aggregate_folds.py \
    --runs "${RUNS[@]}" --splits "${SPLITS[@]}" \
    --name "$PIPE" --out "runs/summary_${MODE}_${PIPE}.json"
