#!/usr/bin/env python
"""Random image-split diagnostic for the inverse structure-to-recipe task.

This is intentionally separate from grouped condition CV. Images from one
condition share the same composition and process labels, so an image-random
split measures interpolation among repeated observations of known conditions.
The reported condition overlap makes that distinction explicit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "scripts")

from tabular_heads import ALLOY, cls_targets, run_fuse


def load_corpus(source_fold: str) -> tuple[np.ndarray, np.ndarray]:
    """Load every image once from one complete grouped fold partition."""
    root = Path("data/conventional") / source_fold
    features, conditions = [], []
    for part in ("train", "val", "test"):
        features.append(np.load(root / f"X_{part}.npy"))
        conditions.extend(json.loads(
            (root / f"conditions_{part}.json").read_text()))
    return np.concatenate(features), np.asarray(conditions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fold", default="loco_fold0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default="runs/summary_random_image_fuse.json")
    args = parser.parse_args()

    features, conditions = load_corpus(args.source_fold)
    strata = np.asarray([ALLOY.loc[c] for c in conditions])
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)

    reports = []
    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(features, strata)):
        X_train, X_test = features[train_idx], features[test_idx]
        train_conditions = conditions[train_idx]
        test_conditions = conditions[test_idx]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        report = run_fuse(
            X_train, X_test, train_conditions, test_conditions)["fuse_argmax"]
        train_set, test_set = set(train_conditions), set(test_conditions)
        overlap = train_set & test_set
        report.update({
            "fold": fold,
            "n_train_images": int(len(train_idx)),
            "n_test_images": int(len(test_idx)),
            "n_test_conditions": len(test_set),
            "condition_overlap_fraction": len(overlap) / len(test_set),
        })
        reports.append(report)
        print(f"fold {fold}: top1={report['alloy_top1']:.3f}  "
              f"WAPE={report['element_wape_present']:.3f}  "
              f"condition_overlap={report['condition_overlap_fraction']:.3f}",
              flush=True)

    metric_keys = [
        "element_wape_present", "mae_elements_macro", "T_ext_wape",
        "alloy_top1", "alloy_top3", "condition_overlap_fraction",
    ]
    summary = {
        "protocol": "random_image_stratified",
        "warning": "Train and test contain images from the same conditions.",
        "n_images": len(features),
        "n_conditions": len(set(conditions)),
        "n_folds": args.folds,
        "metrics_mean": {
            key: float(np.mean([report[key] for report in reports]))
            for key in metric_keys
        },
        "folds": reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics_mean"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()