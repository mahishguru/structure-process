"""kNN retrieval head: soft-vote over nearest training images in the
representation space, with per-pipeline (k, temperature) selection."""

from __future__ import annotations

import numpy as np

from icme_mg.data import (ALLOYS, cls_targets, condition_weights,
                          process_targets)

KNN_GRID = tuple(
    (k, temperature)
    for k in (1, 3, 5, 7)
    for temperature in (0.01, 0.05, 0.1, 0.2))


def knn_outputs(X_train, X_query, train_conditions, balanced=False,
                k=5, temperature=0.05):
    """Return (alloy probabilities, process prediction in transformed space)."""
    train_norm = X_train / (
        np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    query_norm = X_query / (
        np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
    similarity = query_norm @ train_norm.T
    indices = np.argsort(-similarity, axis=1)[:, :k]
    weights = np.take_along_axis(similarity, indices, axis=1)
    weights = np.exp((weights - weights.max(axis=1, keepdims=True))
                     / temperature)
    if balanced:
        weights *= condition_weights(train_conditions)[indices]
    targets = cls_targets(train_conditions)
    probabilities = np.zeros((len(X_query), len(ALLOYS)))
    process = process_targets(train_conditions)
    process_prediction = np.zeros((len(X_query), 2))
    for row in range(len(X_query)):
        for target, weight in zip(targets[indices[row]], weights[row]):
            probabilities[row, target] += weight
        process_prediction[row] = np.average(
            process[indices[row]], axis=0, weights=weights[row])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, process_prediction
