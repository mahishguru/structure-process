#!/usr/bin/env python
"""Final grouped-CV evaluation after refitting on train plus validation.

The fusion hyperparameters are frozen from the completed model study. For each
outer fold, validation conditions can therefore be returned to the fitting set
before the untouched test fold is evaluated. This is the standard final-refit
step and is especially important for alloy classes with four conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

from tabular_heads import run_fuse_balanced


def load_part(fold_dir: Path, part: str):
    features = np.load(fold_dir / f"X_{part}.npy")
    conditions = np.asarray(json.loads(
        (fold_dir / f"conditions_{part}.json").read_text()))
    return features, conditions


def run_fold(fold: int):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train, train_conditions = load_part(fold_dir, "train")
    X_val, val_conditions = load_part(fold_dir, "val")
    X_test, test_conditions = load_part(fold_dir, "test")

    X_fit = np.concatenate([X_train, X_val])
    fit_conditions = np.concatenate([train_conditions, val_conditions])
    report = run_fuse_balanced(
        X_fit, X_test, fit_conditions, test_conditions)[
            "fuse_balanced_argmax"]
    report.update({
        "fold": fold,
        "n_fit_conditions": len(set(fit_conditions)),
        "n_test_conditions": len(set(test_conditions)),
    })
    return report


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: WAPE={report['element_wape_present']:.3f}  "
              f"MAE={report['mae_elements_macro']:.3f}  "
              f"top1={report['alloy_top1']:.3f}", flush=True)

    metrics = ("element_wape_present", "mae_elements_macro", "T_ext_wape",
               "alloy_top1", "alloy_top3")
    summary = {
        "protocol": "grouped_random_cv_final_refit_train_plus_validation",
        "hyperparameters": {
            "catboost_iterations": 500,
            "catboost_depth": 6,
            "catboost_learning_rate": 0.1,
            "knn_k": 5,
            "knn_temperature": 0.05,
            "catboost_knn_weights": [0.5, 0.5],
            "condition_balanced": True,
        },
        "metrics_mean": {
            metric: float(np.mean([report[metric] for report in reports]))
            for metric in metrics
        },
        "folds": reports,
    }
    out = Path("runs/summary_refit_balanced_fusion.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()