#!/usr/bin/env python
"""Scale-aware multiple-instance classifier for multiscale micrographs.

The conventional pipeline contains 90, 100, 120, and 150 um observations for
every condition. Treating them as exchangeable images discards scale identity.
This head concatenates one bootstrap mean descriptor per scale and learns from
condition-level multiscale bags.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")

from tabular_heads import cls_targets
from tune_sparse_fusion import (composition_metrics, full_probabilities,
                                knn_probabilities, scale_subset)
from bootstrap_bag_head import fit_image_classifier

SCALES = (90, 100, 120, 150)
N_BAGS = 32
SAMPLES_PER_SCALE = {90: 8, 100: 4, 120: 2, 150: 1}
SCALE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def load_part(fold_dir: Path, part: str):
    features = np.load(fold_dir / f"X_{part}.npy")
    conditions = np.asarray(json.loads(
        (fold_dir / f"conditions_{part}.json").read_text()))
    stems = json.loads((fold_dir / f"image_stems_{part}.json").read_text())
    scales = np.asarray([int(stem.split("_")[-2]) for stem in stems])
    return features, conditions, scales


def multiscale_bags(features, conditions, scales, seed, n_bags=N_BAGS):
    rng = np.random.default_rng(seed)
    output = []
    output_conditions = []
    for condition in sorted(set(conditions)):
        condition_mask = conditions == condition
        for _ in range(n_bags):
            blocks = []
            for scale in SCALES:
                members = features[condition_mask & (scales == scale)]
                indices = rng.integers(
                    0, len(members), size=SAMPLES_PER_SCALE[scale])
                blocks.append(members[indices].mean(axis=0))
            output.append(np.concatenate(blocks))
            output_conditions.append(condition)
    return np.asarray(output), np.asarray(output_conditions)


def fit_scale_classifier(features, conditions):
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=600, depth=5,
        learning_rate=0.05, l2_leaf_reg=5.0, random_seed=0, verbose=0,
        task_type="CPU", thread_count=-1)
    model.fit(features, cls_targets(conditions))
    return model


def scale_probabilities(model, features, conditions, scales, seed):
    bags, bag_conditions = multiscale_bags(
        features, conditions, scales, seed=seed)
    probabilities = full_probabilities(model, bags)
    pooled = {
        condition: probabilities[bag_conditions == condition].mean(axis=0)
        for condition in sorted(set(conditions))
    }
    return np.stack([pooled[condition] for condition in conditions])


def run_fold(fold):
    fold_dir = Path(f"data/conventional/loco_fold{fold}")
    X_train_raw, train_conditions, train_scales = load_part(fold_dir, "train")
    X_val_raw, val_conditions, val_scales = load_part(fold_dir, "val")
    X_test_raw, test_conditions, test_scales = load_part(fold_dir, "test")
    X_train, X_val, X_test = scale_subset(
        X_train_raw, X_val_raw, X_test_raw,
        columns=np.arange(X_train_raw.shape[1]))

    train_bags, train_bag_conditions = multiscale_bags(
        X_train, train_conditions, train_scales, seed=1000 + fold)
    scale_model = fit_scale_classifier(train_bags, train_bag_conditions)
    image_model = fit_image_classifier(X_train, train_conditions)

    val_scale = scale_probabilities(
        scale_model, X_val, val_conditions, val_scales, seed=2000 + fold)
    val_image = full_probabilities(image_model, X_val)
    val_knn = knn_probabilities(
        X_train, X_val, train_conditions, k=5, temperature=0.05)
    val_baseline = 0.5 * val_image + 0.5 * val_knn
    validation = {
        "image_baseline": composition_metrics(val_baseline, val_conditions)
    }
    for scale_weight in SCALE_WEIGHTS:
        probability = (scale_weight * val_scale
                       + (1.0 - scale_weight) * val_baseline)
        validation[f"scale_{scale_weight:.2f}"] = composition_metrics(
            probability, val_conditions)
    selected = min(
        validation,
        key=lambda name: (validation[name]["element_wape_present"],
                          validation[name]["mae_elements_macro"]))

    test_scale = scale_probabilities(
        scale_model, X_test, test_conditions, test_scales, seed=3000 + fold)
    test_image = full_probabilities(image_model, X_test)
    test_knn = knn_probabilities(
        X_train, X_test, train_conditions, k=5, temperature=0.05)
    test_baseline = 0.5 * test_image + 0.5 * test_knn
    test_variants = {
        "image_baseline": composition_metrics(test_baseline, test_conditions)
    }
    for scale_weight in SCALE_WEIGHTS:
        probability = (scale_weight * test_scale
                       + (1.0 - scale_weight) * test_baseline)
        test_variants[f"scale_{scale_weight:.2f}"] = composition_metrics(
            probability, test_conditions)
    return {
        "fold": fold,
        "selected": selected,
        "validation": validation[selected],
        "test": test_variants[selected],
        "test_variants": test_variants,
    }


def main():
    reports = []
    for fold in range(5):
        report = run_fold(fold)
        reports.append(report)
        print(f"fold {fold}: selected={report['selected']}  "
              f"test={report['test']['element_wape_present']:.3f}  "
              f"baseline={report['test_variants']['image_baseline']['element_wape_present']:.3f}",
              flush=True)
    metrics = ("element_wape_present", "mae_elements_macro",
               "alloy_top1", "alloy_top3")
    names = ("image_baseline",
             *(f"scale_{weight:.2f}" for weight in SCALE_WEIGHTS))
    summary = {
        "protocol": "scale_aware_bootstrap_grouped_random_cv",
        "n_bags": N_BAGS,
        "samples_per_scale": SAMPLES_PER_SCALE,
        "metrics_mean": {
            metric: float(np.mean([
                report["test"][metric] for report in reports]))
            for metric in metrics
        },
        "fixed_variant_means": {
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
    out = Path("runs/summary_scale_aware_bag.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()