#!/usr/bin/env python
"""Condition-level bootstrap-bag classifier for repeated micrographs.

Each condition is represented by bootstrap estimates of descriptor means and
standard deviations. This is label-preserving multiple-instance augmentation:
it exposes finite-image uncertainty without moving synthetic points between
alloy classes or leaking conditions across folds.
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
                                knn_probabilities, load_part, scale_subset)

N_BAGS = 32
BAG_SIZE = 16
BLENDS = {
    "image_baseline": (0.0, 0.5, 0.5),
    "bag_10": (0.1, 0.45, 0.45),
    "bag_20": (0.2, 0.4, 0.4),
    "bag_25": (0.25, 0.375, 0.375),
    "bag_only": (1.0, 0.0, 0.0),
    "bag_direct": (0.5, 0.5, 0.0),
    "bag_knn": (0.5, 0.0, 0.5),
    "bag_heavy": (0.6, 0.2, 0.2),
    "equal": (1 / 3, 1 / 3, 1 / 3),
}
ADAPTIVE_RULES = (
    "disagree_bag50",
    "disagree_bag75",
    "lowconf40_bag50",
    "lowconf50_bag50",
    "lowconf60_bag50",
)
DEPLOYMENT_CHOICES = (
    "image_baseline", "bag_10", "bag_20", "bag_25", *ADAPTIVE_RULES)


def bootstrap_bags(features, conditions, seed, n_bags=N_BAGS):
    rng = np.random.default_rng(seed)
    bag_features = []
    bag_conditions = []
    for condition in sorted(set(conditions)):
        members = features[conditions == condition]
        for _ in range(n_bags):
            indices = rng.integers(0, len(members), size=BAG_SIZE)
            sample = members[indices]
            bag_features.append(np.concatenate([
                sample.mean(axis=0), sample.std(axis=0)]))
            bag_conditions.append(condition)
    return np.asarray(bag_features), np.asarray(bag_conditions)


def fit_image_classifier(features, conditions):
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=6,
        learning_rate=0.1, random_seed=0, verbose=0,
        task_type="GPU", devices="0")
    model.fit(features, cls_targets(conditions),
              sample_weight=condition_weights(conditions))
    return model


def fit_bag_classifier(features, conditions):
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=5,
        learning_rate=0.05, l2_leaf_reg=5.0, random_seed=0, verbose=0,
        task_type="CPU", thread_count=-1)
    model.fit(features, cls_targets(conditions))
    return model


def bag_probabilities(model, features, conditions, seed):
    bags, bag_conditions = bootstrap_bags(features, conditions, seed)
    probabilities = full_probabilities(model, bags)
    condition_probability = {
        condition: probabilities[bag_conditions == condition].mean(axis=0)
        for condition in sorted(set(conditions))
    }
    return np.stack([condition_probability[condition]
                     for condition in conditions])


def adaptive_probabilities(bag, direct, knn, conditions, rule):
    """Use bag evidence only for condition-level ambiguous predictions."""
    output = np.zeros_like(bag)
    for condition in sorted(set(conditions)):
        mask = conditions == condition
        bag_probability = bag[mask].mean(axis=0)
        direct_probability = direct[mask].mean(axis=0)
        knn_probability = knn[mask].mean(axis=0)
        baseline = 0.5 * direct_probability + 0.5 * knn_probability

        if rule.startswith("disagree"):
            use_bag = direct_probability.argmax() != knn_probability.argmax()
        else:
            threshold = int(rule[len("lowconf"):len("lowconf") + 2]) / 100
            use_bag = baseline.max() < threshold
        bag_weight = 0.75 if rule.endswith("bag75") else 0.5
        probability = ((1.0 - bag_weight) * baseline
                       + bag_weight * bag_probability) if use_bag else baseline
        output[mask] = probability
    return output


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions = load_part(fold_dir, "train")
    X_val_raw, val_conditions = load_part(fold_dir, "val")
    X_test_raw, test_conditions = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))

    train_bags, train_bag_conditions = bootstrap_bags(
        X_train, train_conditions, seed=1000 + fold)
    bag_model = fit_bag_classifier(train_bags, train_bag_conditions)
    direct_model = fit_image_classifier(X_train, train_conditions)

    val_sources = (
        bag_probabilities(
            bag_model, X_val, val_conditions, seed=2000 + fold),
        full_probabilities(direct_model, X_val),
        knn_probabilities(
            X_train, X_val, train_conditions, k=5, temperature=0.05),
    )
    validation = {}
    for name, weights in BLENDS.items():
        probability = sum(weight * source
                          for weight, source in zip(weights, val_sources))
        validation[name] = composition_metrics(probability, val_conditions)
    for rule in ADAPTIVE_RULES:
        probability = adaptive_probabilities(
            *val_sources, val_conditions, rule)
        validation[rule] = composition_metrics(probability, val_conditions)
    selected = min(
        DEPLOYMENT_CHOICES,
        key=lambda name: (validation[name]["element_wape_present"],
                          validation[name]["mae_elements_macro"]))

    test_sources = (
        bag_probabilities(
            bag_model, X_test, test_conditions, seed=3000 + fold),
        full_probabilities(direct_model, X_test),
        knn_probabilities(
            X_train, X_test, train_conditions, k=5, temperature=0.05),
    )
    test_blends = {
        name: composition_metrics(
            sum(weight * source
                for weight, source in zip(weights, test_sources)),
            test_conditions)
        for name, weights in BLENDS.items()
    }
    test_blends.update({
        rule: composition_metrics(
            adaptive_probabilities(*test_sources, test_conditions, rule),
            test_conditions)
        for rule in ADAPTIVE_RULES
    })
    return {
        "fold": fold,
        "selected": selected,
        "weights_bag_direct_knn": BLENDS.get(selected),
        "validation": validation[selected],
        "test": test_blends[selected],
        "test_blends": test_blends,
        "test_bag_only": composition_metrics(test_sources[0], test_conditions),
        "test_image_baseline": composition_metrics(
            0.5 * test_sources[1] + 0.5 * test_sources[2], test_conditions),
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: selected={report['selected']}  "
              f"test={report['test']['element_wape_present']:.3f}  "
              f"bag={report['test_bag_only']['element_wape_present']:.3f}  "
              f"baseline={report['test_image_baseline']['element_wape_present']:.3f}",
              flush=True)

    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    summary = {
        "protocol": "bootstrap_bag_grouped_random_cv",
        "n_bags": N_BAGS,
        "bag_size": BAG_SIZE,
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "bag_only_mean": {
            metric: float(np.mean([
                report["test_bag_only"][metric] for report in reports]))
            for metric in metrics
        },
        "image_baseline_mean": {
            metric: float(np.mean([
                report["test_image_baseline"][metric] for report in reports]))
            for metric in metrics
        },
        "fixed_blend_means": {
            name: {
                metric: float(np.mean([
                    report["test_blends"][name][metric]
                    for report in reports]))
                for metric in metrics
            }
            for name in BLENDS
        },
        "adaptive_rule_means": {
            name: {
                metric: float(np.mean([
                    report["test_blends"][name][metric]
                    for report in reports]))
                for metric in metrics
            }
            for name in ADAPTIVE_RULES
        },
        "per_element_wape_mean": {
            element: float(np.mean([
                report["test"]["per_element_wape"][element]
                for report in reports]))
            for element in report["test"]["per_element_wape"]
        },
        "folds": reports,
    }
    out = Path("runs/summary_bootstrap_bag.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()