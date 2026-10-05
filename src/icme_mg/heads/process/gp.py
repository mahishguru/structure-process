"""Gaussian process process head: condition-mean pooled, PCA-reduced, ARD RBF
kernel per target on (T_ext, log v_ext). The stored point prediction applies
the lognormal smearing correction to v_ext; the median is kept for
NLL/coverage so the smeared mean never corrupts the density diagnostics.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import (condition_pool, invert_process, process_targets)
from icme_mg.evaluation.head_metrics import gaussian_uncertainty


def fit_gp_process(X_train, train_conditions):
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, RBF,
                                                   WhiteKernel)

    conditions, pooled = condition_pool(X_train, train_conditions)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0) + 1e-8
    standardized = (pooled - mean) / std
    pca = PCA(n_components=min(32, len(standardized) - 1,
                               standardized.shape[1]), random_state=0)
    reduced = pca.fit_transform(standardized)
    targets = process_targets(conditions)
    models = []
    target_stats = []
    for index in range(2):
        target_mean = targets[:, index].mean()
        target_std = targets[:, index].std() + 1e-8
        # Anisotropic (ARD) RBF: per-dimension lengthscales let the GP learn
        # which principal components matter instead of one shared scale.
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(
                np.full(reduced.shape[1], np.sqrt(reduced.shape[1])),
                length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(0.1, noise_level_bounds=(1e-5, 1e1)),
            normalize_y=False, alpha=1e-6, random_state=0)
        model.fit(reduced, (targets[:, index] - target_mean) / target_std)
        models.append(model)
        target_stats.append((target_mean, target_std))
    return mean, std, pca, models, target_stats


def predict_gp_process(fitted, X_query, query_conditions):
    """Return (conditions, smeared_mean, sigma, uncertainty, median)."""
    mean, std, pca, models, target_stats = fitted
    conditions, pooled = condition_pool(X_query, query_conditions)
    reduced = pca.transform((pooled - mean) / std)
    prediction = np.zeros((len(conditions), 2))
    sigma = np.zeros((len(conditions), 2))
    uncertainty = {}
    truth = process_targets(conditions)
    for index, target in enumerate(("T_ext", "v_ext")):
        normalized_mean, normalized_std = models[index].predict(
            reduced, return_std=True)
        target_mean, target_std = target_stats[index]
        prediction[:, index] = normalized_mean * target_std + target_mean
        predictive_std = np.maximum(normalized_std * target_std, 1e-6)
        sigma[:, index] = predictive_std
        uncertainty[target] = gaussian_uncertainty(
            truth[:, index], prediction[:, index], predictive_std,
            log_target=(target == "v_ext"))
    raw_prediction = invert_process(prediction)
    median_prediction = raw_prediction.copy()
    # Smearing correction: exp(mu) is the lognormal median; the mean is
    # exp(mu + sigma^2/2). Point metrics compare against the mean.
    raw_prediction[:, 1] = np.exp(prediction[:, 1]
                                  + 0.5 * np.square(sigma[:, 1]))
    return conditions, raw_prediction, sigma, uncertainty, median_prediction
