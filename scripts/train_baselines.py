#!/usr/bin/env python
"""Classical per-label baselines (XGBoost, GP) on the conventional descriptor
matrices. Instance-level fit, condition-level evaluation (mean over instances).

Example:
    python scripts/train_baselines.py --split loco_fold0 --kind xgboost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from icme_mg import LABEL_ORDER
from icme_mg.models.baselines import fit_classical, predict_classical
from icme_mg.evaluation.metrics import aggregate_per_condition, per_label_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--kind", default="xgboost", choices=["xgboost", "gp"])
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    fold_dir = data_root / "conventional" / args.split
    labels = pd.read_csv(data_root / "labels" / "labels.csv"
                         ).set_index("condition_id")

    def load_part(part):
        X = np.load(fold_dir / f"X_{part}.npy")
        cids = json.loads((fold_dir / f"conditions_{part}.json").read_text())
        Y = labels.loc[cids, LABEL_ORDER].to_numpy(float)
        return X, Y, cids

    X_tr, Y_tr, _ = load_part("train")
    X_te, Y_te, cids_te = load_part("test")

    models = fit_classical(X_tr, Y_tr, kind=args.kind,
                           seed=cfg["classical"]["seed"])
    pred = predict_classical(models, X_te)

    cond_ids, pred_c = aggregate_per_condition(cids_te, pred)
    _, true_c = aggregate_per_condition(cids_te, Y_te)
    metrics = per_label_metrics(true_c, pred_c)

    out_dir = Path(args.out or f"runs/classical_{args.kind}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "predictions.npz", condition_ids=cond_ids,
             y_pred=pred_c, y_true=true_c)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
