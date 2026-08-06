#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SPLIT="random_seed0"
CHAIN_LOG="runs/cross_pipeline_${SPLIT}_chain.log"
mkdir -p runs data/gnn/embeddings

exec > >(tee -a "$CHAIN_LOG") 2>&1

printf '[%s] preparing labels and split\n' "$(date --iso-8601=seconds)"
# Rebuilding these needs the raw image database on the external drive; reuse
# the cached artifacts when they are already present.
if [[ -f data/labels/labels.csv ]]; then
    printf '[%s] reusing data/labels/labels.csv\n' "$(date --iso-8601=seconds)"
else
    "$PYTHON" scripts/build_labels.py --config configs/paths.yaml
fi

if [[ -f "data/splits/${SPLIT}.json" ]]; then
    printf '[%s] reusing data/splits/%s.json\n' \
        "$(date --iso-8601=seconds)" "$SPLIT"
else
    "$PYTHON" scripts/build_splits.py --config configs/paths.yaml
fi

printf '[%s] starting CPU descriptor build and GPU GNN training\n' \
    "$(date --iso-8601=seconds)"
descriptor_pid=""
if [[ -f "data/conventional/${SPLIT}/X_test.npy" ]]; then
    printf '[%s] reusing conventional descriptors\n' \
        "$(date --iso-8601=seconds)"
else
    "$PYTHON" scripts/build_conventional.py --config configs/conventional.yaml &
    descriptor_pid=$!
fi

if [[ ! -f "runs/gnn_${SPLIT}_s90/best.pt" ]]; then
    "$PYTHON" scripts/train.py \
        --pipeline gnn \
        --split "$SPLIT" \
        --scales 90 \
        --out "runs/gnn_${SPLIT}_s90"
else
    printf '[%s] reusing GNN checkpoint\n' "$(date --iso-8601=seconds)"
fi

if [[ -n "$descriptor_pid" ]]; then
    wait "$descriptor_pid"
fi

failed=()
for pipeline in conventional genai gnn; do
    output="runs/summary_cross_pipeline_heads_${pipeline}_${SPLIT}.json"
    store="runs/predictions_cross_pipeline_${pipeline}_${SPLIT}.npz"
    if [[ -f "$output" && -f "$store" ]]; then
        printf '[%s] reusing %s\n' "$(date --iso-8601=seconds)" "$output"
        continue
    fi
    printf '[%s] benchmarking %s\n' \
        "$(date --iso-8601=seconds)" "$pipeline"
    # One pipeline failure must not cancel the remaining pipelines.
    if ! "$PYTHON" scripts/cross_pipeline_heads.py \
        --pipeline "$pipeline" \
        --split "$SPLIT" \
        --out "$output" \
        --predictions "$store"; then
        printf '[%s] FAILED %s\n' "$(date --iso-8601=seconds)" "$pipeline"
        failed+=("$pipeline")
    fi
done

if (( ${#failed[@]} )); then
    printf '[%s] completed with failures: %s\n' \
        "$(date --iso-8601=seconds)" "${failed[*]}"
    exit 1
fi

printf '[%s] all random-seed-0 experiments complete\n' \
    "$(date --iso-8601=seconds)"