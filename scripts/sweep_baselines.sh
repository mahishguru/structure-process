#!/usr/bin/env bash
# Hyperparameter sweep for the conventional and genai pipelines on dev70.
# Same flow_ar head as the GNN; only training hyperparameters are swept.
# Selection on best per-condition val NLL; the winner is evaluated on test.
# Resume-safe: skips any config whose result.json already exists.
set -u
PY=.venv/bin/python
COMMON="--split dev70 --epochs 400 --device cuda"

# tag  lr     wd    spc
CFGS=(
  "lr1e3      1e-3  1e-4  16"
  "lr1e4      1e-4  1e-4  16"
  "wd1e3      3e-4  1e-3  16"
  "spc32      3e-4  1e-4  32"
  "lr1e3wd1e3 1e-3  1e-3  16"
)

for pipe in conventional genai; do
    for cfg in "${CFGS[@]}"; do
        read -r tag lr wd spc <<< "$cfg"
        out="runs/sweep_${pipe}_${tag}"
        if [[ -f "$out/result.json" ]]; then
            echo "[sweep] $pipe/$tag done, skipping"
            continue
        fi
        echo "[sweep] === $pipe $tag (lr=$lr wd=$wd spc=$spc) ==="
        $PY -u scripts/train.py --pipeline "$pipe" $COMMON \
            --lr "$lr" --weight-decay "$wd" --samples-per-condition "$spc" \
            --out "$out" \
            || { echo "[sweep] $pipe/$tag FAILED"; exit 1; }
    done
done

echo "[sweep] all trained; selecting winners"
$PY - <<'EOF'
import json
from pathlib import Path

tags = ["lr1e3", "lr1e4", "wd1e3", "spc32", "lr1e3wd1e3"]
base = {"conventional": "runs/conventional_dev70", "genai": "runs/genai_dev70"}
winners = {}
for pipe in ("conventional", "genai"):
    rows = [("base(3e-4,1e-4,16)", base[pipe])]
    rows += [(t, f"runs/sweep_{pipe}_{t}") for t in tags]
    print(f"\n== {pipe} (best val NLL) ==")
    best = None
    for tag, d in rows:
        try:
            v = json.loads(Path(d, "result.json").read_text())["best_val_nll"]
        except FileNotFoundError:
            print(f"  {tag:22} MISSING"); continue
        print(f"  {tag:22} {v:8.3f}")
        if best is None or v < best[1]:
            best = (d, v, tag)
    winners[pipe] = best
    print(f"  -> winner: {best[2]} ({best[1]:.3f})")
Path("runs/sweep_winners.json").write_text(json.dumps(
    {p: {"run": w[0], "val_nll": w[1], "tag": w[2]}
     for p, w in winners.items()}, indent=2))
EOF

for pipe in conventional genai; do
    run=$($PY -c "import json;print(json.load(open('runs/sweep_winners.json'))['$pipe']['run'])")
    if [[ ! -f "$run/eval_dev70.json" ]]; then
        echo "[sweep] evaluating winner $run"
        $PY -u scripts/evaluate.py --run "$run" --pipeline "$pipe" \
            --split dev70 --device cuda \
            || { echo "[sweep] eval FAILED for $run"; exit 1; }
    fi
    echo "=== $pipe winner: $run ==="
    $PY -c "
import json
r = json.load(open('$run/eval_dev70.json'))
print('joint_nll', round(r['joint_nll'],3), 'presF1', round(r['presence_f1']['macro'],3),
      'top1', round(r['alloy_topk']['top1'],2), 'top3', round(r['alloy_topk']['top3'],2),
      'ece', round(r['quantile_ece']['macro'],3))"
done
echo "[sweep] done $(date)"
