#!/usr/bin/env bash
# Detached chain: build conventional dev70 features, then train+eval the
# conventional and genai pipelines on dev70 (same flow_ar head as the GNN,
# only the encoder/adapter differs). Resume-safe per stage.
set -u
PY=.venv/bin/python
COMMON="--split dev70 --epochs 400 --samples-per-condition 16 --device cuda"

echo "[chain] start $(date)"

# 1. conventional dev70 features (skip if already built)
if [[ ! -f data/conventional/dev70/X_train.npy ]]; then
    echo "[chain] building conventional dev70 features"
    $PY -u scripts/build_conventional.py configs/conventional_dev70.yaml \
        || { echo "[chain] conventional build FAILED"; exit 1; }
fi

# 2. conventional pipeline
out=runs/conventional_dev70
if [[ ! -f $out/eval_dev70.json ]]; then
    echo "[chain] training conventional"
    $PY -u scripts/train.py --pipeline conventional $COMMON --out $out \
        || { echo "[chain] conventional train FAILED"; exit 1; }
    echo "[chain] evaluating conventional"
    $PY -u scripts/evaluate.py --run $out --pipeline conventional \
        --split dev70 --device cuda \
        || { echo "[chain] conventional eval FAILED"; exit 1; }
fi

# 3. genai pipeline
out=runs/genai_dev70
if [[ ! -f $out/eval_dev70.json ]]; then
    echo "[chain] training genai"
    $PY -u scripts/train.py --pipeline genai $COMMON --out $out \
        || { echo "[chain] genai train FAILED"; exit 1; }
    echo "[chain] evaluating genai"
    $PY -u scripts/evaluate.py --run $out --pipeline genai \
        --split dev70 --device cuda \
        || { echo "[chain] genai eval FAILED"; exit 1; }
fi

echo "[chain] done $(date)"
for d in runs/conventional_dev70 runs/genai_dev70; do
    echo "=== $d ==="
    $PY -c "
import json
r = json.load(open('$d/eval_dev70.json'))
print('joint_nll', round(r['joint_nll'], 3),
      'presF1', round(r['presence_f1']['macro'], 3),
      'top1', round(r['alloy_topk']['top1'], 2),
      'top3', round(r['alloy_topk']['top3'], 2))"
done
