#!/usr/bin/env python
"""Validation-tuned sparse-data fusion on grouped random condition folds.

Each outer fold uses only its training and validation conditions to select a
descriptor subset, kNN settings, and CatBoost/kNN blend. The selected pipeline
is evaluated once on the untouched test conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

from tabular_heads import (ALLOYS, ALLOY, ELEM, LB, NOMINAL,
                           condition_weights)


FEATURE_SETS = {
    "all_438": np.arange(438),
    "acta_380": np.r_[0:320, 378:438],
    "threepoint_gsh": np.r_[0:120, 378:438],
    "gram_gsh": np.r_[120:320, 378:438],
    "shape_gsh": np.arange(320, 438),
    "gsh": np.arange(378, 438),
}
K_VALUES = (3, 5, 9, 15)
TEMPERATURES = (0.02, 0.05, 0.10)
FUSION_WEIGHTS = (0.25, 0.50, 0.75)


def load_part(fold_dir: Path, part: str):
    features = np.load(fold_dir / f"X_{part}.npy")
    conditions = np.asarray(json.loads(
        (fold_dir / f"conditions_{part}.json").read_text()))
    return features, conditions


def scale_subset(X_train, *others, columns):
    train = X_train[:, columns]
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return ((train - mean) / std,
            *((other[:, columns] - mean) / std for other in others))


def class_targets(condition_ids):
    return np.asarray([ALLOYS.index(ALLOY.loc[c]) for c in condition_ids])


def fit_catboost(X_train, train_conditions, X_val, val_conditions):
    from catboost import CatBoostClassifier, Pool

    train_pool = Pool(
        X_train, class_targets(train_conditions),
        weight=condition_weights(train_conditions))
    val_pool = Pool(
        X_val, class_targets(val_conditions),
        weight=condition_weights(val_conditions))
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=1000, depth=6,
        learning_rate=0.05, l2_leaf_reg=5.0, random_seed=0,
        verbose=0, task_type="GPU", devices="0",
        use_best_model=True, early_stopping_rounds=80)
    model.fit(train_pool, eval_set=val_pool)
    return model


def full_probabilities(model, features):
    probabilities = np.zeros((len(features), len(ALLOYS)))
    probabilities[:, [int(c) for c in model.classes_]] = \
        np.asarray(model.predict_proba(features))
    return probabilities


def knn_probabilities(X_train, X_query, train_conditions, k, temperature):
    train_norm = X_train / (
        np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    query_norm = X_query / (
        np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
    similarity = query_norm @ train_norm.T
    indices = np.argsort(-similarity, axis=1)[:, :k]
    weights = np.take_along_axis(similarity, indices, axis=1)
    weights = np.exp((weights - weights.max(axis=1, keepdims=True))
                     / temperature)
    weights *= condition_weights(train_conditions)[indices]
    targets = class_targets(train_conditions)
    probabilities = np.zeros((len(X_query), len(ALLOYS)))
    for row in range(len(X_query)):
        for target, weight in zip(targets[indices[row]], weights[row]):
            probabilities[row, target] += weight
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def condition_probabilities(probabilities, condition_ids):
    conditions = sorted(set(condition_ids))
    pooled = np.stack([
        probabilities[condition_ids == condition].mean(axis=0)
        for condition in conditions
    ])
    return conditions, pooled


def composition_metrics(probabilities, condition_ids):
    conditions, pooled = condition_probabilities(probabilities, condition_ids)
    order = np.argsort(-pooled, axis=1)
    prediction = NOMINAL[order[:, 0]]
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    per_element = {}
    wapes = []
    for index, element in enumerate(ELEM):
        present = truth[:, index] > 0
        if present.any():
            value = float(np.abs(
                truth[present, index] - prediction[present, index]).sum()
                / truth[present, index].sum())
            per_element[element] = value
            wapes.append(value)
    targets = class_targets(conditions)
    return {
        "element_wape_present": float(np.mean(wapes)),
        "mae_elements_macro": float(np.abs(truth - prediction).mean()),
        "alloy_top1": float(np.mean(order[:, 0] == targets)),
        "alloy_top3": float(np.mean([
            targets[row] in order[row, :3] for row in range(len(targets))])),
        "per_element_wape": per_element,
    }


def selection_key(report):
    return report["element_wape_present"], report["mae_elements_macro"]


def run_fold(fold: int):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")

    candidates = {}
    for name, columns in FEATURE_SETS.items():
        X_train, X_val, X_test = scale_subset(
            X_train_raw, X_val_raw, X_test_raw, columns=columns)
        model = fit_catboost(
            X_train, train_conditions, X_val, val_conditions)
        val_prob = full_probabilities(model, X_val)
        candidates[name] = {
            "columns": columns, "X_train": X_train, "X_val": X_val,
            "X_test": X_test, "model": model, "val_cat": val_prob,
            "val_report": composition_metrics(val_prob, val_conditions),
        }
    feature_name = min(
        candidates, key=lambda name: selection_key(
            candidates[name]["val_report"]))
    chosen = candidates[feature_name]

    knn_candidates = []
    for k in K_VALUES:
        for temperature in TEMPERATURES:
            probability = knn_probabilities(
                chosen["X_train"], chosen["X_val"], train_conditions,
                k, temperature)
            report = composition_metrics(probability, val_conditions)
            knn_candidates.append((selection_key(report), k, temperature,
                                   probability, report))
    _, k, temperature, val_knn, val_knn_report = min(
        knn_candidates, key=lambda item: item[0])

    fusion_candidates = []
    for cat_weight in FUSION_WEIGHTS:
        probability = cat_weight * chosen["val_cat"] + \
            (1.0 - cat_weight) * val_knn
        report = composition_metrics(probability, val_conditions)
        fusion_candidates.append((selection_key(report), cat_weight, report))
    _, cat_weight, val_report = min(
        fusion_candidates, key=lambda item: item[0])

    test_cat = full_probabilities(chosen["model"], chosen["X_test"])
    test_knn = knn_probabilities(
        chosen["X_train"], chosen["X_test"], train_conditions,
        k, temperature)
    test_probability = cat_weight * test_cat + (1.0 - cat_weight) * test_knn
    test_report = composition_metrics(test_probability, test_conditions)
    return {
        "fold": fold,
        "selected": {
            "features": feature_name, "k": k, "temperature": temperature,
            "cat_weight": cat_weight,
            "cat_iterations": chosen["model"].get_best_iteration() + 1,
        },
        "validation": val_report,
        "validation_cat": chosen["val_report"],
        "validation_knn": val_knn_report,
        "test": test_report,
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: {report['selected']}  "
              f"val={report['validation']['element_wape_present']:.3f}  "
              f"test={report['test']['element_wape_present']:.3f}",
              flush=True)

    metric_names = (
        "element_wape_present", "mae_elements_macro", "alloy_top1",
        "alloy_top3")
    summary = {
        "protocol": "validation_tuned_grouped_random_cv",
        "metrics_mean": {
            name: float(np.mean([report["test"][name] for report in reports]))
            for name in metric_names
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
    out = Path("runs/summary_tuned_sparse_fusion.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics_mean"], indent=2))
    print(json.dumps(summary["per_element_wape_mean"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()