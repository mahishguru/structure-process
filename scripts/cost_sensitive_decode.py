#!/usr/bin/env python
"""Cost-sensitive decoding for the condition-balanced fusion posterior.

Argmax minimizes alloy 0-1 loss, whereas the headline metric is an asymmetric
present-element WAPE. This script estimates an alloy confusion cost matrix from
training conditions and selects the posterior temperature, fusion weight, and
cost/0-1 tradeoff on validation conditions. Predictions remain restricted to
the 14 observed, physically valid alloy recipes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")

from tabular_heads import (ALLOYS, ALLOY, ELEM, LB, NOMINAL, cls_targets,
                           condition_weights)
from tune_sparse_fusion import (condition_probabilities, full_probabilities,
                                knn_probabilities, load_part, scale_subset)

FUSION_WEIGHTS = (0.25, 0.5, 0.75)
TEMPERATURES = (0.5, 1.0, 2.0)
COST_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_classifier(features, conditions):
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=6,
        learning_rate=0.1, random_seed=0, verbose=0,
        task_type="GPU", devices="0")
    model.fit(features, cls_targets(conditions),
              sample_weight=condition_weights(conditions))
    return model


def composition_cost(train_conditions):
    """Estimate per-condition contribution to macro present-element WAPE."""
    unique_conditions = sorted(set(train_conditions))
    truth = np.stack([
        LB.loc[condition, ELEM].to_numpy(float)
        for condition in unique_conditions
    ])
    denominator_rate = truth.sum(axis=0) / len(truth)
    denominator_rate = np.maximum(denominator_rate, 1e-8)

    cost = np.zeros((len(ALLOYS), len(ALLOYS)))
    for true_index, true_composition in enumerate(NOMINAL):
        present = true_composition > 0
        for predicted_index, predicted_composition in enumerate(NOMINAL):
            contributions = np.zeros(len(ELEM))
            contributions[present] = (
                np.abs(true_composition[present]
                       - predicted_composition[present])
                / denominator_rate[present])
            cost[true_index, predicted_index] = contributions.mean()
    positive = cost[cost > 0]
    return cost / np.median(positive)


def temper(probabilities, temperature):
    adjusted = np.power(np.clip(probabilities, 1e-12, 1.0), 1 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def decode(probabilities, cost, cost_weight):
    zero_one = np.ones_like(cost) - np.eye(len(cost))
    decision_cost = cost_weight * cost + (1.0 - cost_weight) * zero_one
    expected_cost = probabilities @ decision_cost
    return np.argsort(expected_cost, axis=1)


def evaluate(probabilities, condition_ids, cost, cost_weight):
    conditions, pooled = condition_probabilities(probabilities, condition_ids)
    order = decode(pooled, cost, cost_weight)
    prediction = NOMINAL[order[:, 0]]
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    per_element = {}
    wapes = []
    for index, element in enumerate(ELEM):
        present = truth[:, index] > 0
        value = float(np.abs(
            truth[present, index] - prediction[present, index]).sum()
            / truth[present, index].sum())
        per_element[element] = value
        wapes.append(value)
    targets = cls_targets(conditions)
    return {
        "element_wape_present": float(np.mean(wapes)),
        "mae_elements_macro": float(np.abs(truth - prediction).mean()),
        "alloy_top1": float(np.mean(order[:, 0] == targets)),
        "alloy_top3": float(np.mean([
            targets[row] in order[row, :3] for row in range(len(targets))])),
        "per_element_wape": per_element,
    }


def selection_key(report):
    """Prioritize WAPE, rejecting large MAE degradation as a tie breaker."""
    return report["element_wape_present"], report["mae_elements_macro"]


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))

    model = fit_classifier(X_train, train_conditions)
    val_cat = full_probabilities(model, X_val)
    val_knn = knn_probabilities(
        X_train, X_val, train_conditions, k=5, temperature=0.05)
    cost = composition_cost(train_conditions)

    candidates = []
    for fusion_weight in FUSION_WEIGHTS:
        fused = fusion_weight * val_cat + (1 - fusion_weight) * val_knn
        for temperature in TEMPERATURES:
            probability = temper(fused, temperature)
            for cost_weight in COST_WEIGHTS:
                report = evaluate(
                    probability, val_conditions, cost, cost_weight)
                candidates.append((selection_key(report), fusion_weight,
                                   temperature, cost_weight, report))
    _, fusion_weight, temperature, cost_weight, val_report = min(
        candidates, key=lambda item: item[0])

    test_cat = full_probabilities(model, X_test)
    test_knn = knn_probabilities(
        X_train, X_test, train_conditions, k=5, temperature=0.05)
    test_fused = fusion_weight * test_cat + (1 - fusion_weight) * test_knn
    test_report = evaluate(
        temper(test_fused, temperature), test_conditions, cost, cost_weight)
    baseline_report = evaluate(
        0.5 * test_cat + 0.5 * test_knn, test_conditions,
        np.ones_like(cost) - np.eye(len(cost)), cost_weight=1.0)
    return {
        "fold": fold,
        "selected": {
            "cat_weight": fusion_weight,
            "posterior_temperature": temperature,
            "composition_cost_weight": cost_weight,
        },
        "validation": val_report,
        "test": test_report,
        "test_argmax_baseline": baseline_report,
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: selected={report['selected']}  "
              f"test={report['test']['element_wape_present']:.3f}  "
              f"baseline={report['test_argmax_baseline']['element_wape_present']:.3f}",
              flush=True)

    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    summary = {
        "protocol": "validation_tuned_cost_sensitive_grouped_random_cv",
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "argmax_baseline_mean": {
            metric: float(np.mean([
                report["test_argmax_baseline"][metric] for report in reports]))
            for metric in metrics
        },
        "per_element_wape_mean": {
            element: float(np.mean([
                report["test"]["per_element_wape"][element]
                for report in reports]))
            for element in ELEM
        },
        "folds": reports,
    }
    out = Path("runs/summary_cost_sensitive_decode.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()