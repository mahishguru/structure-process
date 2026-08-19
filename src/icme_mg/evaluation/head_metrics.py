"""Metrics for the prediction-head benchmark (numpy, condition-level).

Every composition report carries the guard metrics alongside the headline
present-WAPE, so a head cannot trade false-positive elements for a better
headline number without it showing up in the same row:

- element_wape_present:  WAPE over elements actually present (the deployment
  question: the alloy family is known at optimization time, only the levels
  are needed).
- element_wape_all:      WAPE over all lattice elements, absent ones included.
- macro_false_positive_rate: fraction of absent elements predicted present.

Decoded composition comes from the alloy lattice: argmax (or regressed vector
snapped to the nearest nominal) -> nominal wt% of that alloy.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import (ALLOYS, ELEM, LB, NOMINAL, cls_targets,
                          condition_pool)

PRESENCE_THRESHOLD = 1e-8
Z90 = 1.6448536269514722


def presence_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Per-element WAPE (present and all) plus presence confusion rates.

    truth/prediction: (n_conditions, n_elements) in wt%.
    """
    predicted_present = prediction > PRESENCE_THRESHOLD
    per_element = {}
    present_wapes, all_wapes, fp_rates = [], [], []
    for index, element in enumerate(ELEM):
        actual_present = truth[:, index] > 0
        denominator = max(float(truth[actual_present, index].sum()), 1e-8)
        present_wape = float(
            np.abs(truth[actual_present, index]
                   - prediction[actual_present, index]).sum() / denominator)
        all_wape = float(
            np.abs(truth[:, index] - prediction[:, index]).sum()
            / denominator)
        absent = ~actual_present
        fp_rate = float(
            predicted_present[absent, index].mean()) if absent.any() else 0.0
        tp = int(np.sum(predicted_present[:, index] & actual_present))
        fp = int(np.sum(predicted_present[:, index] & absent))
        fn = int(np.sum(~predicted_present[:, index] & actual_present))
        per_element[element] = {
            "wape_present": present_wape,
            "wape_all": all_wape,
            "mae": float(np.abs(
                truth[:, index] - prediction[:, index]).mean()),
            "presence_precision": tp / (tp + fp) if tp + fp else 0.0,
            "presence_recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp_rate,
        }
        present_wapes.append(present_wape)
        all_wapes.append(all_wape)
        fp_rates.append(fp_rate)
    return {
        "element_wape_present": float(np.mean(present_wapes)),
        "element_wape_all": float(np.mean(all_wapes)),
        "element_mae_macro": float(np.abs(truth - prediction).mean()),
        "macro_false_positive_rate": float(np.mean(fp_rates)),
        "per_element": per_element,
    }


def composition_metrics(probabilities: np.ndarray,
                        condition_ids: np.ndarray) -> dict:
    """Classifier decode: pool image probabilities, decode top-1 alloy."""
    conditions, pooled = condition_pool(probabilities, condition_ids)
    order = np.argsort(-pooled, axis=1)
    prediction = NOMINAL[order[:, 0]]
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    report = presence_metrics(truth, prediction)
    targets = cls_targets(conditions)
    report.update({
        "alloy_top1": float(np.mean(order[:, 0] == targets)),
        "alloy_top3": float(np.mean([
            targets[row] in order[row, :3]
            for row in range(len(targets))])),
        "class_nll": float(-np.log(np.clip(
            pooled[np.arange(len(targets)), targets], 1e-12, 1)).mean()),
    })
    return report


def regression_composition_metrics(predictions: np.ndarray,
                                   condition_ids: np.ndarray) -> dict:
    """Regressor decode: pool element predictions, snap to nearest nominal
    for the alloy ranking."""
    conditions, prediction = condition_pool(predictions, condition_ids)
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    report = presence_metrics(truth, prediction)
    mean, std = NOMINAL.mean(0), NOMINAL.std(0) + 1e-8
    distance = np.linalg.norm(
        ((prediction - mean) / std)[:, None]
        - ((NOMINAL - mean) / std)[None], axis=2)
    order = distance.argsort(1)
    targets = cls_targets(conditions)
    report.update({
        "alloy_top1": float(np.mean(order[:, 0] == targets)),
        "alloy_top3": float(np.mean([
            targets[row] in order[row, :3]
            for row in range(len(targets))])),
    })
    return report


def process_metrics(predictions: np.ndarray,
                    condition_ids: np.ndarray) -> dict:
    """Point metrics for (T_ext, v_ext) in physical units."""
    conditions, prediction = condition_pool(predictions, condition_ids)
    truth = LB.loc[conditions, ["T_ext", "v_ext"]].to_numpy(float)
    return condition_level_process_metrics(prediction, conditions, truth)


def condition_level_process_metrics(prediction: np.ndarray, conditions,
                                    truth: np.ndarray | None = None) -> dict:
    if truth is None:
        truth = LB.loc[conditions, ["T_ext", "v_ext"]].to_numpy(float)
    output = {}
    for index, target in enumerate(("T_ext", "v_ext")):
        residual = truth[:, index] - prediction[:, index]
        ss_total = np.square(truth[:, index] - truth[:, index].mean()).sum()
        output[target] = {
            "mae": float(np.abs(residual).mean()),
            "mape": float(np.abs(residual / truth[:, index]).mean()),
            "wape": float(np.abs(residual).sum()
                          / np.abs(truth[:, index]).sum()),
            "r2": float(1 - np.square(residual).sum()
                        / max(ss_total, 1e-12)),
        }
    return output


def gaussian_uncertainty(truth: np.ndarray, mean: np.ndarray,
                         sigma: np.ndarray, log_target: bool) -> dict:
    """NLL and 90% coverage for a Gaussian predictive; log_target applies the
    lognormal Jacobian so the NLL stays a density in physical units."""
    sigma = np.maximum(sigma, 1e-6)
    z_score = (truth - mean) / sigma
    jacobian = float(truth.mean()) if log_target else 0.0
    return {
        "nll": float(np.mean(
            0.5 * np.square(z_score) + np.log(sigma)
            + 0.5 * np.log(2 * np.pi)) + jacobian),
        "coverage90": float(np.mean(np.abs(z_score) < Z90)),
    }
