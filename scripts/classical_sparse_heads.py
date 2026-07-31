#!/usr/bin/env python
"""Classical high-dimensional classifiers for the scarce-condition regime.

RBF/linear SVM and shrinkage LDA provide smooth, strongly regularized decision
boundaries complementary to boosted trees. Hyperparameters and posterior
fusion weights are selected only on each outer fold's validation conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

sys.path.insert(0, "scripts")

from tabular_heads import cls_targets, condition_weights
from tune_sparse_fusion import (composition_metrics, full_probabilities,
                                knn_probabilities, load_part, scale_subset)


def model_candidates():
    for kernel in ("linear", "rbf"):
        for c_value in (0.1, 1.0, 10.0):
            yield f"svm_{kernel}_c{c_value:g}", SVC(
                C=c_value, kernel=kernel, gamma="scale",
                probability=True, random_state=0, cache_size=4096)
    yield "lda_shrinkage", LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto")


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))
    targets = cls_targets(train_conditions)
    weights = condition_weights(train_conditions)
    val_knn = knn_probabilities(
        X_train, X_val, train_conditions, k=5, temperature=0.05)

    candidates = []
    fitted = {}
    for name, model in model_candidates():
        if name.startswith("svm"):
            model.fit(X_train, targets, sample_weight=weights)
        else:
            # Equal-size deterministic subsampling approximates condition
            # weights for estimators without sample_weight support.
            rng = np.random.default_rng(0)
            indices = np.concatenate([
                rng.choice(np.flatnonzero(train_conditions == condition),
                           size=16, replace=True)
                for condition in sorted(set(train_conditions))
            ])
            model.fit(X_train[indices], targets[indices])
        fitted[name] = model
        val_model = full_probabilities(model, X_val)
        for model_weight in (0.5, 0.75, 1.0):
            probability = (model_weight * val_model
                           + (1.0 - model_weight) * val_knn)
            report = composition_metrics(probability, val_conditions)
            candidates.append((report["element_wape_present"],
                               report["mae_elements_macro"], name,
                               model_weight, report))
    _, _, selected, model_weight, val_report = min(candidates)

    test_model = full_probabilities(fitted[selected], X_test)
    test_knn = knn_probabilities(
        X_train, X_test, train_conditions, k=5, temperature=0.05)
    test_probability = (model_weight * test_model
                        + (1.0 - model_weight) * test_knn)
    return {
        "fold": fold,
        "selected": selected,
        "model_weight": model_weight,
        "validation": val_report,
        "test": composition_metrics(test_probability, test_conditions),
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: {report['selected']}  "
              f"weight={report['model_weight']:.2f}  "
              f"test={report['test']['element_wape_present']:.3f}",
              flush=True)
    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    summary = {
        "protocol": "validation_tuned_classical_grouped_random_cv",
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "per_element_wape_mean": {
            element: float(np.mean([
                report["test"]["per_element_wape"][element]
                for report in reports]))
            for element in reports[0]["test"]["per_element_wape"]
        },
        "folds": reports,
    }
    out = Path("runs/summary_classical_sparse_heads.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()