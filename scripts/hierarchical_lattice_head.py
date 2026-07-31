#!/usr/bin/env python
"""Hierarchical classifier for the sparse Mg-Gd-Mn composition lattice.

The nine Mg-Gd-Mn alloys form a complete 3x3 lattice but four corners have
only four conditions each. Instead of learning every corner independently,
this head predicts Gd family membership, Gd level, and Mn level. The two level
classifiers pool evidence across lattice rows and columns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")

from tabular_heads import (ALLOYS, ALLOY, ELEM, NOMINAL, cls_targets,
                           condition_weights)
from tune_sparse_fusion import (composition_metrics, full_probabilities,
                                knn_probabilities, load_part, scale_subset)

GD_INDEX = ELEM.index("Gd")
MN_INDEX = ELEM.index("Mn")
GD_ALLOYS = np.flatnonzero(NOMINAL[:, GD_INDEX] > 0)
OTHER_ALLOYS = np.flatnonzero(NOMINAL[:, GD_INDEX] == 0)
GD_LEVELS = np.unique(NOMINAL[GD_ALLOYS, GD_INDEX])
MN_LEVELS = np.unique(NOMINAL[GD_ALLOYS, MN_INDEX])

BLENDS = {
    "hierarchical": (1.0, 0.0, 0.0),
    "direct_knn": (0.0, 0.5, 0.5),
    "hierarchical_direct": (0.5, 0.5, 0.0),
    "hierarchical_knn": (0.5, 0.0, 0.5),
    "hierarchical_heavy": (0.6, 0.2, 0.2),
    "equal": (1 / 3, 1 / 3, 1 / 3),
}


def fit_classifier(features, targets, conditions):
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=6,
        learning_rate=0.1, l2_leaf_reg=3.0, random_seed=0, verbose=0,
        task_type="GPU", devices="0")
    model.fit(features, targets, sample_weight=condition_weights(conditions))
    return model


def expanded_probabilities(model, features, classes):
    output = np.zeros((len(features), len(classes)))
    output[:, [int(value) for value in model.classes_]] = \
        np.asarray(model.predict_proba(features))
    return output


def fit_hierarchy(X_train, train_conditions):
    alloy_targets = cls_targets(train_conditions)
    is_gd = np.isin(alloy_targets, GD_ALLOYS)

    other_mapping = {alloy: index + 1
                     for index, alloy in enumerate(OTHER_ALLOYS)}
    family_targets = np.asarray([
        0 if gd else other_mapping[alloy]
        for alloy, gd in zip(alloy_targets, is_gd)
    ])
    family = fit_classifier(X_train, family_targets, train_conditions)

    gd_amount = NOMINAL[alloy_targets[is_gd], GD_INDEX]
    mn_amount = NOMINAL[alloy_targets[is_gd], MN_INDEX]
    gd_targets = np.searchsorted(GD_LEVELS, gd_amount)
    mn_targets = np.searchsorted(MN_LEVELS, mn_amount)
    gd_level = fit_classifier(
        X_train[is_gd], gd_targets, train_conditions[is_gd])
    mn_level = fit_classifier(
        X_train[is_gd], mn_targets, train_conditions[is_gd])
    return family, gd_level, mn_level


def hierarchy_probabilities(models, features):
    family, gd_level, mn_level = models
    family_probability = expanded_probabilities(
        family, features, np.arange(1 + len(OTHER_ALLOYS)))
    gd_probability = expanded_probabilities(
        gd_level, features, np.arange(len(GD_LEVELS)))
    mn_probability = expanded_probabilities(
        mn_level, features, np.arange(len(MN_LEVELS)))

    output = np.zeros((len(features), len(ALLOYS)))
    for offset, alloy in enumerate(OTHER_ALLOYS, start=1):
        output[:, alloy] = family_probability[:, offset]
    for alloy in GD_ALLOYS:
        gd_index = int(np.searchsorted(
            GD_LEVELS, NOMINAL[alloy, GD_INDEX]))
        mn_index = int(np.searchsorted(
            MN_LEVELS, NOMINAL[alloy, MN_INDEX]))
        output[:, alloy] = (family_probability[:, 0]
                            * gd_probability[:, gd_index]
                            * mn_probability[:, mn_index])
    return output / output.sum(axis=1, keepdims=True)


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw, columns=np.arange(438))

    hierarchy = fit_hierarchy(X_train, train_conditions)
    direct = fit_classifier(
        X_train, cls_targets(train_conditions), train_conditions)

    val_hierarchical = hierarchy_probabilities(hierarchy, X_val)
    val_direct = full_probabilities(direct, X_val)
    val_knn = knn_probabilities(
        X_train, X_val, train_conditions, k=5, temperature=0.05)
    val_sources = (val_hierarchical, val_direct, val_knn)

    validation = {}
    for name, weights in BLENDS.items():
        probability = sum(weight * source
                          for weight, source in zip(weights, val_sources))
        validation[name] = composition_metrics(probability, val_conditions)
    selected = min(
        validation,
        key=lambda name: (validation[name]["element_wape_present"],
                          validation[name]["mae_elements_macro"]))

    test_sources = (
        hierarchy_probabilities(hierarchy, X_test),
        full_probabilities(direct, X_test),
        knn_probabilities(
            X_train, X_test, train_conditions, k=5, temperature=0.05),
    )
    test_probability = sum(
        weight * source
        for weight, source in zip(BLENDS[selected], test_sources))
    return {
        "fold": fold,
        "selected": selected,
        "weights_hierarchical_direct_knn": BLENDS[selected],
        "validation": validation[selected],
        "test": composition_metrics(test_probability, test_conditions),
        "test_hierarchical": composition_metrics(
            test_sources[0], test_conditions),
        "test_direct_knn": composition_metrics(
            0.5 * test_sources[1] + 0.5 * test_sources[2], test_conditions),
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: selected={report['selected']}  "
              f"test={report['test']['element_wape_present']:.3f}  "
              f"hier={report['test_hierarchical']['element_wape_present']:.3f}  "
              f"baseline={report['test_direct_knn']['element_wape_present']:.3f}",
              flush=True)

    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    summary = {
        "protocol": "hierarchical_lattice_grouped_random_cv",
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "hierarchical_only_mean": {
            metric: float(np.mean([
                report["test_hierarchical"][metric] for report in reports]))
            for metric in metrics
        },
        "direct_knn_mean": {
            metric: float(np.mean([
                report["test_direct_knn"][metric] for report in reports]))
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
    out = Path("runs/summary_hierarchical_lattice.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()