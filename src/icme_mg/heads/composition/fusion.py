"""Condition-balanced CatBoost + kNN fusion: the owner head.

Equal total training weight per condition in both components, 50:50
probability blend. CatBoost depth/iterations/learning-rate are selected on
the fold's validation part; kNN (k, temperature) likewise.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import ALLOYS, cls_targets, condition_weights
from icme_mg.heads.composition.knn import KNN_GRID, knn_outputs

CAT_GRID = (
    {"iterations": 500, "depth": 6, "learning_rate": 0.1},
    {"iterations": 900, "depth": 4, "learning_rate": 0.05},
    {"iterations": 500, "depth": 8, "learning_rate": 0.05},
)


def fit_catboost(X_train, train_conditions, balanced=False,
                 iterations=500, depth=6, learning_rate=0.1):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=iterations, depth=depth,
        learning_rate=learning_rate, random_seed=0, verbose=0,
        task_type="GPU", devices="0")
    weights = condition_weights(train_conditions) if balanced else None
    model.fit(X_train, cls_targets(train_conditions), sample_weight=weights)
    return model


def full_probabilities(model, features) -> np.ndarray:
    output = np.zeros((len(features), len(ALLOYS)))
    output[:, [int(value) for value in model.classes_]] = np.asarray(
        model.predict_proba(features))
    return output


def fit_balanced_fusion(X_train, train_conditions,
                        cat_params=None, knn_params=(5, 0.05)):
    """Owner model: balanced CatBoost plus the training set for balanced kNN
    retrieval."""
    cat_params = dict(cat_params or {})
    model = fit_catboost(X_train, train_conditions, balanced=True,
                         **cat_params)
    return {
        "catboost": model,
        "X_train": X_train,
        "train_conditions": np.asarray(train_conditions),
        "knn_params": tuple(knn_params),
    }


def fusion_probabilities(model, X_query) -> np.ndarray:
    k, temperature = model["knn_params"]
    catboost = full_probabilities(model["catboost"], X_query)
    knn, _ = knn_outputs(
        model["X_train"], X_query, model["train_conditions"],
        balanced=True, k=k, temperature=temperature)
    return 0.5 * catboost + 0.5 * knn


def tune_catboost_classifier(X_train, train_conditions, X_val,
                             val_conditions, score_fn):
    """Select CAT_GRID params on the validation part. score_fn maps
    (probabilities, condition_ids) to a scalar to minimize."""
    best = None
    for params in CAT_GRID:
        model = fit_catboost(X_train, train_conditions, **params)
        score = score_fn(full_probabilities(model, X_val), val_conditions)
        if best is None or score < best[0]:
            best = (score, params)
    return dict(best[1])


def tune_knn(X_train, train_conditions, X_val, val_conditions, score_fn):
    """Select (k, temperature) on the validation part."""
    best = None
    for k, temperature in KNN_GRID:
        probabilities, _ = knn_outputs(
            X_train, X_val, train_conditions, k=k, temperature=temperature)
        score = score_fn(probabilities, val_conditions)
        if best is None or score < best[0]:
            best = (score, (k, temperature))
    return best[1]
