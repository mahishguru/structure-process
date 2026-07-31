#!/usr/bin/env bash
# Waits for run5 (GNN control) and the baselines chain, then prints a combined
# dev70 comparison across pipelines and GNN variants.
set -u
while pgrep -f "run5_noaug_gpu" > /dev/null || \
      pgrep -f "chain_dev70_baselines.sh" > /dev/null; do
    sleep 60
done
echo "[compare] all jobs finished at $(date)"
.venv/bin/python - <<'EOF'
import json
import numpy as np

rows = [
    ("gnn run2 (cpu, no aug)",  "runs/gnn_dev70_s90_e400"),
    ("gnn run4a (aug .05)",     "runs/gnn_dev70_s90_run4a_aug05"),
    ("gnn run4b (aug+wd)",      "runs/gnn_dev70_s90_run4b_aug05_wd1e3"),
    ("gnn run5 (gpu, no aug)",  "runs/gnn_dev70_s90_run5_noaug_gpu"),
    ("conventional",            "runs/conventional_dev70"),
    ("genai",                   "runs/genai_dev70"),
]
hdr = (f"{'pipeline':26} {'jointNLL':>8} {'presF1':>7} {'top1':>5} "
       f"{'top3':>5} {'ece':>6} {'cov90':>6} {'elMAE':>6} {'T_MAE':>6}")
print(hdr)
for name, d in rows:
    try:
        r = json.load(open(f"{d}/eval_dev70.json"))
    except FileNotFoundError:
        print(f"{name:26} MISSING")
        continue
    cov = np.mean(list(r["coverage_90"].values()))
    el = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd", "Y"]
    elmae = np.mean([r["per_label"][e]["mae"] for e in el])
    print(f"{name:26} {r['joint_nll']:8.3f} {r['presence_f1']['macro']:7.3f} "
          f"{r['alloy_topk']['top1']:5.2f} {r['alloy_topk']['top3']:5.2f} "
          f"{r['quantile_ece']['macro']:6.3f} {cov:6.2f} {elmae:6.3f} "
          f"{r['per_label']['T_ext']['mae']:6.1f}")
EOF
