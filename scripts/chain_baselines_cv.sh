#!/usr/bin/env bash
# Chain 1: wait for the baselines sweep to finish, then run LOCO (and LOAO)
# CV for conventional and genai using each pipeline's winning hyperparameters.
set -u
PY=.venv/bin/python

echo "[chain] waiting for runs/sweep_winners.json ..."
until [[ -f runs/sweep_winners.json ]]; do sleep 60; done
echo "[chain] sweep winners found $(date)"

hp() {  # tag -> "lr wd spc"
    case "$1" in
        base*)        echo "3e-4 1e-4 16" ;;
        lr1e3wd1e3)   echo "1e-3 1e-3 16" ;;
        lr1e3)        echo "1e-3 1e-4 16" ;;
        lr1e4)        echo "1e-4 1e-4 16" ;;
        wd1e3)        echo "3e-4 1e-3 16" ;;
        spc32)        echo "3e-4 1e-4 32" ;;
        *) echo "3e-4 1e-4 16" ;;
    esac
}

for pipe in conventional genai; do
    tag=$($PY -c "import json;print(json.load(open('runs/sweep_winners.json'))['$pipe']['tag'])")
    read -r lr wd spc <<< "$(hp "$tag")"
    echo "[chain] $pipe winner tag=$tag -> lr=$lr wd=$wd spc=$spc"
    bash scripts/run_cv_pipe.sh "$pipe" loco "$lr" "$wd" "$spc" \
        || { echo "[chain] $pipe LOCO FAILED"; exit 1; }
    bash scripts/run_cv_pipe.sh "$pipe" loao "$lr" "$wd" "$spc" \
        || { echo "[chain] $pipe LOAO FAILED"; exit 1; }
done
echo "[chain] baselines CV all done $(date)"
