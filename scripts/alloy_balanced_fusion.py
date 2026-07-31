#!/usr/bin/env python
"""Condition- and alloy-balanced fusion for sparse alloy classes.

Condition balancing removes image-count bias but still gives an 18-condition
alloy 4.5x the training mass of a four-condition alloy. This script applies a
power-law correction to alloy frequency in both CatBoost loss weights and kNN
votes. Beta=0 is the current model; beta=1 gives every alloy equal total mass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")

from tabular_heads import ALLOYS, ALLOY, cls_targets, condition_weights
from tune_sparse_fusion import (composition_metrics, full_probabilities,
                                load_part, scale_subset)

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def training_weights(condition_ids, beta):
    base = condition_weights(condition_ids)
    unique_conditions = sorted(set(condition_ids))
    condition_alloys = np.asarray([ALLOY.loc[c] for c in unique_conditions])
    alloys, counts = np.unique(condition_alloys, return_counts=True)
    count = dict(zip(alloys, counts))
    correction = np.asarray([
        count[ALLOY.loc[condition]] ** (-beta)
        for condition in condition_ids
    ])
    weights = base * correction
    return weights / weights.mean()


def knn_with_weights(X_train, X_query, train_conditions, sample_weight,
                     k=5, temperature=0.05):
    train_norm = X_train / (
        np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    query_norm = X_query / (
        np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
    similarity = query_norm @ train_norm.T
    indices = np.argsort(-similarity, axis=1)[:, :k]
    weights = np.take_along_axis(similarity, indices, axis=1)
    weights = np.exp((weights - weights.max(axis=1, keepdims=True))
                     / temperature)
    weights *= sample_weight[indices]
    targets = cls_targets(train_conditions)
    probabilities = np.zeros((len(X_query), len(ALLOYS)))
    for row in range(len(X_query)):
        for target, weight in zip(targets[indices[row]], weights[row]):
            probabilities[row, target] += weight
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_predict(X_train, train_conditions, queries, beta):
    weights = training_weights(train_conditions, beta)
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=6,
        learning_rate=0.1, random_seed=0, verbose=0,
        task_type="GPU", devices="0")
    model.fit(X_train, cls_targets(train_conditions), sample_weight=weights)
    output = []
    for X_query in queries:
        cat = full_probabilities(model, X_query)
        knn = knn_with_weights(
            X_train, X_query, train_conditions, weights)
        output.append(0.5 * cat + 0.5 * knn)
    return output


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))

    validation = {}
    test = {}
    for beta in BETAS:
        name = f"beta_{beta:.2f}"
        val_probability, test_probability = fit_predict(
            X_train, train_conditions, (X_val, X_test), beta)
        validation[name] = composition_metrics(
            val_probability, val_conditions)
        test[name] = composition_metrics(
            test_probability, test_conditions)
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
        print(f"fold {fold}: selected={report['selected']}  "
              f"test={report['test']['element_wape_present']:.3f}  "
              f"baseline={report['test_variants']['beta_0.00']['element_wape_present']:.3f}",
              flush=True)
    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    names = tuple(f"beta_{beta:.2f}" for beta in BETAS)
    summary = {
        "protocol": "alloy_balanced_grouped_random_cv",
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "fixed_beta_means": {
            name: {
                metric: float(np.mean([
                    report["test_variants"][name][metric]
                    for report in reports]))
                for metric in metrics
            }
            for name in names
        },
        "folds": reports,
    }
    out = Path("runs/summary_alloy_balanced_fusion.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()