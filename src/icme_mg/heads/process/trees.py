"""Gradient-boosted tree process head: CatBoost MultiRMSE on standardized
(T_ext, log v_ext), with validation-selected hyperparameters."""

from __future__ import annotations

import numpy as np

from icme_mg.data import invert_process, process_targets

CAT_GRID = (
    {"iterations": 500, "depth": 6, "learning_rate": 0.1},
    {"iterations": 900, "depth": 4, "learning_rate": 0.05},
    {"iterations": 500, "depth": 8, "learning_rate": 0.05},
)


def fit_catboost_process(X_train, train_conditions,
                         iterations=500, depth=6, learning_rate=0.1):
    from catboost import CatBoostRegressor

    targets = process_targets(train_conditions)
    mean = targets.mean(axis=0)
    std = targets.std(axis=0) + 1e-8
    model = CatBoostRegressor(
        loss_function="MultiRMSE", iterations=iterations, depth=depth,
        learning_rate=learning_rate, random_seed=0, verbose=0,
        boosting_type="Plain", task_type="GPU", devices="0")
    # MultiRMSE sums raw squared errors across targets, so unstandardized
    # targets would give v_ext 0.1% of the loss.
    model.fit(X_train, (targets - mean) / std)
    return model, mean, std


def predict_catboost_process(fitted, features) -> np.ndarray:
    model, mean, std = fitted
    return invert_process(np.asarray(model.predict(features)) * std + mean)


def tune_catboost_process(X_train, train_conditions, X_val, val_conditions,
                          score_fn):
    """Select CAT_GRID params on the validation part. score_fn maps a
    physical-unit prediction and condition ids to a scalar to minimize."""
    best = None
    for params in CAT_GRID:
        fitted = fit_catboost_process(X_train, train_conditions, **params)
        score = score_fn(predict_catboost_process(fitted, X_val),
                         val_conditions)
        if best is None or score < best[0]:
            best = (score, params, fitted)
    return best[2], dict(best[1])
