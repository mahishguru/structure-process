"""Evaluation protocol: 5-fold condition-grouped cross-validation.

The folds are the condition-grouped partitions in data/splits/cv_fold0..4.
Each fold provides train/val/test condition lists; val is used only for
base-hyperparameter selection, test is touched once.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import ALLOY

CV_FOLDS = tuple(f"cv_fold{index}" for index in range(5))


def condition_stratified_splits(conditions, requested: int, seed: int):
    """Inner CV over conditions. Stratified by alloy when every alloy has at
    least `requested` conditions (so stratification does not collapse the
    fold count); plain shuffled KFold otherwise.

    The 5-fold outer partitions leave some Mg-Gd-Mn alloys with only two
    training conditions, where the classic min-class rule would degrade to a
    2-fold split and a 50% holdout; condition-level KFold keeps the requested
    granularity. Condition grouping is what the OOF honesty requires; alloy
    stratification inside inner folds is best-effort.

    Returns (splits, protocol_name, n_folds) with splits a list of
    (fit_indices, held_indices) into the given condition order.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    conditions = [str(c) for c in conditions]
    y = np.array([ALLOY.loc[c] for c in conditions])
    _, counts = np.unique(y, return_counts=True)
    if int(counts.min()) >= requested:
        splitter = StratifiedKFold(
            n_splits=requested, shuffle=True, random_state=seed)
        return (list(splitter.split(np.zeros(len(y)), y)),
                "stratified_condition_kfold", requested)
    folds = min(requested, len(y))
    if folds < 2:
        raise ValueError("At least two training conditions are required")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return (list(splitter.split(np.zeros(len(y)))),
            "condition_kfold_unstratified_sparse_alloys", folds)
