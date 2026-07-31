#!/usr/bin/env python
"""Validation-selected long-tail correction for balanced fusion posteriors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

from bootstrap_bag_head import fit_image_classifier
from tabular_heads import ALLOYS, ALLOY, ELEM
from tune_sparse_fusion import (composition_metrics, full_probabilities,
                                knn_probabilities, load_part, scale_subset)

TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)


def alloy_condition_counts(condition_ids):
    counts = {alloy: 0 for alloy in ALLOYS}
    for condition in set(condition_ids):
        counts[ALLOY.loc[condition]] += 1
    return np.asarray([counts[alloy] for alloy in ALLOYS], dtype=float)


def adjust_prior(probabilities, counts, tau):
    """Remove a power of the empirical condition-level alloy prior."""
    adjusted = probabilities / np.power(counts, tau)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_validation_raw, validation_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_validation, X_test = scale_subset(
        X_train_raw, X_validation_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))

    model = fit_image_classifier(X_train, train_conditions)
    validation_probability = (
        0.5 * full_probabilities(model, X_validation)
        + 0.5 * knn_probabilities(
            X_train, X_validation, train_conditions,
            k=5, temperature=0.05))
    test_probability = (
        0.5 * full_probabilities(model, X_test)
        + 0.5 * knn_probabilities(
            X_train, X_test, train_conditions, k=5, temperature=0.05))
    counts = alloy_condition_counts(train_conditions)

    validation = {}
    test = {}
    for tau in TAUS:
        name = f"tau_{tau:.2f}"
        validation[name] = composition_metrics(
            adjust_prior(validation_probability, counts, tau),
            validation_conditions)
        test[name] = composition_metrics(
            adjust_prior(test_probability, counts, tau), test_conditions)
    selected = min(
        validation,
        key=lambda name: (validation[name]["element_wape_present"],
                          validation[name]["mae_elements_macro"]))
    return {
        "fold": fold,
        "selected": selected,
        "validation": validation[selected],
        "test": test[selected],
        "test_variants": test,
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(
            f"fold {fold}: selected={report['selected']}  "
            f"test={report['test']['element_wape_present']:.3f}  "
            f"baseline={report['test_variants']['tau_0.00']['element_wape_present']:.3f}",
            flush=True)

    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    names = tuple(f"tau_{tau:.2f}" for tau in TAUS)
    summary = {
        "protocol": "validation_selected_logit_adjusted_grouped_random_cv",
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "fixed_tau_means": {
            name: {
                metric: float(np.mean([
                    report["test_variants"][name][metric]
                    for report in reports]))
                for metric in metrics
            }
            for name in names
        },
        "per_element_wape_mean": {
            element: float(np.mean([
                report["test"]["per_element_wape"][element]
                for report in reports
                if element in report["test"]["per_element_wape"]]))
            for element in ELEM
        },
        "folds": reports,
    }
    out = Path("runs/summary_logit_adjusted_fusion.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()