"""Condition-distribution joint-grid GP for both Task B targets.

Representation-agnostic: each condition is a bag of representation rows. The
feature map is a PCA-compressed, RBF-sampled condition-distribution embedding
blended 50:50 (kernel weight) with the known composition + extrusion-ratio
kernel.

Both targets share one intrinsic-correlation joint GP over (T_ext, log v_ext)
and one posterior over the discrete grid of observed training (T, v) pairs;
the process space is a small discrete catalogue, not a continuum.

T_ext: minimum-risk decode over that grid under a joint temperature/velocity
loss.

v_ext: the same catalogue constraint applied to the other axis. A continuous
estimate (a bounded log-space GP, optionally blended with an ordinal
classifier over velocity levels or with the grid posterior median) is snapped
onto the observed velocity levels, nearest in log space. Extrusion speed is
set from a short menu of press settings rather than dialled continuously, so
a value between two settings is never a possible answer; which continuous
estimate feeds the snap is chosen on training OOF predictions under safety
constraints (MAE and WAPE must not degrade, R2 must not drop vs the plain
snapped log-GP, which is always eligible).

Every distribution transform is refitted inside the velocity OOF folds.
Validation and test select nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from icme_mg.data import ALLOY, ELEM, LB, ratio_features
from icme_mg.evaluation.head_metrics import (condition_level_process_metrics,
                                             gaussian_uncertainty)
from icme_mg.protocols.cv import condition_stratified_splits

Z90 = 1.6448536269514722


@dataclass
class GridGPConfig:
    pca_components: int = 8
    gamma_factor: float = 1.0
    known_weight: float = 0.5
    rff_components: int = 256
    samples_per_condition: int = 64
    distance_sample: int = 1200
    gp_noise: float = 0.05
    variance_temperature: float = 2.0
    inner_folds: int = 5
    penalty_grid: tuple = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    blend_grid: tuple = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    grid_temperature_grid: tuple = (0.5, 1.0, 2.0, 4.0)
    seed: int = 0


# ------------------------------------------------------------------------ bags
def build_bags(raw_parts) -> dict:
    """One bag per condition: representation rows, known composition + ratio,
    true process."""
    output = {}
    for part, (features, condition_ids) in zip(("train", "val", "test"),
                                               raw_parts):
        features = np.asarray(features, dtype=float)
        condition_ids = np.asarray(condition_ids).astype(str)
        bags = []
        for condition in sorted(set(condition_ids)):
            composition = LB.loc[condition, ELEM].to_numpy(float)
            ratio = ratio_features(np.asarray([condition]))[0]
            bags.append({
                "condition": condition,
                "alloy": str(ALLOY.loc[condition]),
                "images": features[condition_ids == condition],
                "known": np.concatenate([composition, ratio]),
                "truth": LB.loc[condition, ["T_ext", "v_ext"]].to_numpy(float),
            })
        output[part] = bags
    return output


# ------------------------------------------------------------- feature mapping
def fit_mapper(bags, config: GridGPConfig, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.metrics import pairwise_distances

    means = np.stack([bag["images"].mean(axis=0) for bag in bags])
    seconds = np.stack([
        np.square(bag["images"]).mean(axis=0) for bag in bags])
    mean = means.mean(axis=0)
    scale = np.sqrt(np.maximum(
        seconds.mean(axis=0) - np.square(mean), 1e-12))
    rng = np.random.default_rng(seed)
    samples = []
    for bag in bags:
        values = bag["images"]
        if len(values) > config.samples_per_condition:
            values = values[rng.choice(
                len(values), config.samples_per_condition, replace=False)]
        samples.append(values)
    sample = (np.concatenate(samples) - mean) / scale
    width = min(config.pca_components, len(sample) - 1, sample.shape[1])
    pca = PCA(n_components=width, random_state=seed).fit(sample)
    projected = pca.transform(sample)
    if len(projected) > config.distance_sample:
        projected_distance = projected[rng.choice(
            len(projected), config.distance_sample, replace=False)]
    else:
        projected_distance = projected
    distances = pairwise_distances(
        projected_distance, metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[upper > 1e-12]
    median = float(np.median(upper)) if len(upper) else 1.0
    rff = RBFSampler(
        gamma=config.gamma_factor / median,
        n_components=config.rff_components, random_state=seed).fit(projected)
    known = np.stack([bag["known"] for bag in bags])
    known_mean = known.mean(axis=0)
    known_scale = known.std(axis=0) + 1e-8
    known_scaled = (known - known_mean) / known_scale
    known_distance = pairwise_distances(known_scaled, metric="sqeuclidean")
    known_upper = known_distance[np.triu_indices_from(known_distance, k=1)]
    known_upper = known_upper[known_upper > 1e-12]
    known_median = float(np.median(known_upper)) if len(known_upper) else 1.0
    return {
        "mean": mean, "scale": scale, "pca": pca, "rff": rff,
        "known_mean": known_mean, "known_scale": known_scale,
        "known_gamma": 1.0 / known_median,
    }


def transform(mapper, bags):
    embeddings = []
    for bag in bags:
        values = (bag["images"] - mapper["mean"]) / mapper["scale"]
        values = mapper["rff"].transform(mapper["pca"].transform(values))
        embeddings.append(values.mean(axis=0))
    embeddings = np.stack(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    known = np.stack([bag["known"] for bag in bags])
    known = (known - mapper["known_mean"]) / mapper["known_scale"]
    return embeddings, known


def kernel(left, right, mapper, weight):
    from sklearn.metrics import pairwise_distances

    distribution_kernel = left[0] @ right[0].T
    known_kernel = np.exp(-mapper["known_gamma"] * pairwise_distances(
        left[1], right[1], metric="sqeuclidean"))
    return ((1.0 - weight) * distribution_kernel
            + weight * known_kernel)


def _solve(matrix, values):
    jitter = 1e-8
    for _ in range(8):
        try:
            factor = np.linalg.cholesky(
                matrix + jitter * np.eye(len(matrix)))
            alpha = np.linalg.solve(
                factor.T, np.linalg.solve(factor, values))
            return factor, alpha
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("kernel matrix is not positive definite")


def _latent(bags):
    truth = np.stack([bag["truth"] for bag in bags])
    return np.column_stack([truth[:, 0], np.log(truth[:, 1])])


# ------------------------------------------------------------------ joint GP
def fit_joint(kernel_matrix, bags, config: GridGPConfig):
    latent = _latent(bags)
    mean = latent.mean(axis=0)
    scale = latent.std(axis=0) + 1e-8
    normalized = (latent - mean) / scale
    correlation = float(np.clip(
        np.corrcoef(normalized.T)[0, 1], -0.9, 0.9))
    task = np.asarray([[1.0, correlation], [correlation, 1.0]])
    joint = np.einsum("ij,ab->iajb", kernel_matrix, task).reshape(
        2 * len(kernel_matrix), 2 * len(kernel_matrix))
    factor, alpha = _solve(
        joint + config.gp_noise * np.eye(len(joint)), normalized.reshape(-1))
    return {"factor": factor, "alpha": alpha, "mean": mean,
            "scale": scale, "task": task, "noise": config.gp_noise}


def predict_joint(model, cross):
    """Per-query latent Gaussian: means (n, 2) and covariances (n, 2, 2) in
    (T_ext, log v_ext) units."""
    blocks = np.einsum(
        "qj,ab->qajb", cross, model["task"]).reshape(len(cross), 2, -1)
    means, covariances = [], []
    for block in blocks:
        normalized = block @ model["alpha"]
        solved = np.linalg.solve(model["factor"], block.T)
        covariance = model["task"] - solved.T @ solved
        covariance = (covariance + covariance.T) / 2
        covariance.flat[::3] = np.maximum(np.diag(covariance), 1e-10)
        means.append(normalized * model["scale"] + model["mean"])
        covariances.append(
            covariance * np.outer(model["scale"], model["scale"]))
    return np.stack(means), np.stack(covariances)


def process_grid(train_bags):
    """The discrete catalogue of (T, v) pairs actually run, with counts."""
    return np.unique(np.stack([bag["truth"] for bag in train_bags]),
                     axis=0, return_counts=True)


def grid_posterior(model, means, covariances, grid, counts,
                   temperature: float):
    """Posterior over the observed process catalogue for each query.

    `temperature` inflates the latent covariance before the catalogue is
    scored; it trades a peaked posterior (trusts the GP) against a flat one
    (falls back on how often each recipe was run).
    """
    from scipy.special import logsumexp

    latent_grid = np.column_stack([grid[:, 0], np.log(grid[:, 1])])
    prior = np.log((counts + 1.0) / (counts.sum() + len(counts)))
    noise = model["noise"] * np.diag(np.square(model["scale"]))
    posterior = []
    for mean, covariance in zip(means, covariances):
        covariance = temperature * covariance + noise + 1e-8 * np.eye(2)
        delta = latent_grid - mean
        log_probability = (
            -0.5 * np.einsum(
                "ij,jk,ik->i", delta, np.linalg.inv(covariance), delta)
            + prior)
        posterior.append(np.exp(
            log_probability - logsumexp(log_probability)))
    return np.stack(posterior)


def grid_temperature(probability, grid, temperature_scale: float):
    """Minimum-risk decode over the observed training (T, v) grid."""
    predictions = []
    for row in probability:
        risks = np.asarray([
            np.sum(row * (
                np.abs(grid[:, 0] - candidate[0]) / temperature_scale
                + np.abs(grid[:, 1] - candidate[1]) / grid[:, 1]))
            for candidate in grid
        ])
        predictions.append(grid[np.argmin(risks), 0])
    return np.asarray(predictions)


def grid_velocity(probability, grid):
    """Posterior median velocity over the observed catalogue.

    The median, not the arg max: with 20-30 catalogue entries the arg max is
    a brittle winner-take-all pick, whereas the median is the minimum-risk
    point under absolute error and degrades gracefully when the posterior is
    spread over several plausible recipes.
    """
    order = np.argsort(grid[:, 1])
    velocity = grid[order, 1]
    cumulative = np.cumsum(probability[:, order], axis=1)
    index = np.argmax(cumulative >= 0.5 * cumulative[:, -1:], axis=1)
    return velocity[index]


# ------------------------------------------------------------- velocity heads
def fit_log_gp(kernel_matrix, bags, config: GridGPConfig):
    target = np.log([bag["truth"][1] for bag in bags])
    mean, scale = float(target.mean()), float(target.std() + 1e-8)
    factor, alpha = _solve(
        kernel_matrix + config.gp_noise * np.eye(len(kernel_matrix)),
        (target - mean) / scale)
    bounds = np.asarray([bag["truth"][1] for bag in bags])
    return factor, alpha, mean, scale, bounds.min(), bounds.max()


def predict_log_gp(fitted, cross):
    _, alpha, mean, scale, minimum, maximum = fitted
    return np.clip(np.exp(cross @ alpha * scale + mean), minimum, maximum)


def _vector(mapped, weight):
    return np.concatenate([
        np.sqrt(1.0 - weight) * mapped[0],
        np.sqrt(weight) * mapped[1],
    ], axis=1)


def fit_ordinal(features, velocity, config: GridGPConfig, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    levels = np.unique(velocity)
    models = []
    for index, threshold in enumerate(levels[:-1]):
        target = (velocity > threshold).astype(int)
        if np.unique(target).size == 1:
            models.append(float(target[0]))
            continue
        model = HistGradientBoostingClassifier(
            learning_rate=0.07, max_leaf_nodes=3, min_samples_leaf=6,
            l2_regularization=1.0, max_iter=300,
            random_state=seed + index).fit(features, target)
        models.append(model)
    return levels, models


def ordinal_probabilities(fitted, features):
    levels, models = fitted
    if len(levels) == 1:
        return np.ones((len(features), 1))
    survival = []
    for model in models:
        if isinstance(model, float):
            survival.append(np.full(len(features), model))
        else:
            index = int(np.flatnonzero(model.classes_ == 1)[0])
            survival.append(model.predict_proba(features)[:, index])
    survival = np.minimum.accumulate(np.column_stack(survival), axis=1)
    probabilities = np.column_stack([
        1 - survival[:, 0], survival[:, :-1] - survival[:, 1:],
        survival[:, -1],
    ])
    probabilities = np.maximum(probabilities, 0)
    return probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-12)


def risk_velocity(levels, probabilities, gp, mean_velocity, penalty):
    loss = np.abs(levels[:, None] - levels[None, :]) / levels[None, :]
    posterior = probabilities @ loss.T
    guard = np.abs(levels[None, :] - gp[:, None]) / mean_velocity
    return levels[np.argmin(posterior + penalty * guard, axis=1)]


def snap_levels(values, levels):
    """Project onto the observed velocity catalogue, nearest in log space.

    Extrusion speed is a press setting chosen from a short menu (11 distinct
    values across the 107 conditions), so a value between two settings is
    never right. Log space is the correct metric because the menu is roughly
    geometric, spanning 0.5 to 7.5 mm/s.
    """
    levels = np.asarray(levels, dtype=float)
    distance = np.abs(np.log(levels)[None, :]
                      - np.log(np.maximum(values, 1e-8))[:, None])
    return levels[np.argmin(distance, axis=1)]


def _snap_oof(values, level_sets):
    """Snap each OOF prediction onto the catalogue its own inner fold saw."""
    return np.asarray([
        snap_levels(values[index:index + 1], level_sets[index])[0]
        for index in range(len(values))])


def _target_metrics(truth, prediction):
    residual = truth - prediction
    total = np.square(truth - truth.mean()).sum()
    return {
        "mae": float(np.abs(residual).mean()),
        "mape": float(np.abs(residual / truth).mean()),
        "wape": float(np.abs(residual).sum() / truth.sum()),
        "r2": float(1 - np.square(residual).sum() / max(total, 1e-12)),
        "log_mae": float(np.abs(
            np.log(truth) - np.log(np.maximum(prediction, 1e-8))).mean()),
    }


def select_velocity(train_bags, config: GridGPConfig, log):
    """OOF selection of the velocity decoder under safety constraints; the
    pure log-GP fallback at blend=0 is always eligible."""
    conditions = [bag["condition"] for bag in train_bags]
    splits, protocol, folds = condition_stratified_splits(
        conditions, config.inner_folds, config.seed)
    truth = np.asarray([bag["truth"][1] for bag in train_bags])
    gp_oof = np.empty(len(train_bags))
    level_sets = [None] * len(train_bags)
    risk_oof = {
        penalty: np.empty(len(train_bags))
        for penalty in config.penalty_grid}
    grid_oof = {
        temperature: np.empty(len(train_bags))
        for temperature in config.grid_temperature_grid}
    for fold, (fit_index, held_index) in enumerate(splits):
        fitted_bags = [train_bags[index] for index in fit_index]
        held_bags = [train_bags[index] for index in held_index]
        mapper = fit_mapper(fitted_bags, config, config.seed + 1009 * fold)
        mapped_fit = transform(mapper, fitted_bags)
        mapped_held = transform(mapper, held_bags)
        fit_kernel = kernel(mapped_fit, mapped_fit, mapper,
                            config.known_weight)
        cross = kernel(mapped_held, mapped_fit, mapper, config.known_weight)
        gp = predict_log_gp(
            fit_log_gp(fit_kernel, fitted_bags, config), cross)
        fit_velocity = np.asarray(
            [bag["truth"][1] for bag in fitted_bags])
        ordinal = fit_ordinal(
            _vector(mapped_fit, config.known_weight),
            fit_velocity, config, config.seed + fold)
        probabilities = ordinal_probabilities(
            ordinal, _vector(mapped_held, config.known_weight))
        gp_oof[held_index] = gp
        levels = np.unique(fit_velocity)
        for index in held_index:
            level_sets[index] = levels
        for penalty in config.penalty_grid:
            risk_oof[penalty][held_index] = risk_velocity(
                ordinal[0], probabilities, gp, fit_velocity.mean(), penalty)
        # The grid decoder needs the joint GP, so it is refitted here too:
        # the catalogue seen by the decoder must be the inner-fold catalogue.
        joint = fit_joint(fit_kernel, fitted_bags, config)
        means, covariances = predict_joint(joint, cross)
        grid, counts = process_grid(fitted_bags)
        for temperature in config.grid_temperature_grid:
            grid_oof[temperature][held_index] = grid_velocity(
                grid_posterior(joint, means, covariances, grid, counts,
                               temperature), grid)
        log(f"    velocity OOF fold {fold + 1}/{len(splits)}")
    reference = _target_metrics(truth, _snap_oof(gp_oof, level_sets))

    def evaluate(prediction, **fields):
        report = _target_metrics(truth, _snap_oof(prediction, level_sets))
        safe = (report["mae"] <= reference["mae"]
                and report["wape"] <= reference["wape"]
                and report["r2"] >= reference["r2"])
        return {"decoder": "log_gp", "penalty": 0.0, "blend": 0.0,
                "grid_temperature": None, "safe": safe,
                **fields, **report}

    rows = [evaluate(gp_oof)]
    rows[0]["safe"] = True
    candidates = [
        ("ordinal", "penalty", penalty, values)
        for penalty, values in risk_oof.items()]
    candidates += [
        ("grid_median", "grid_temperature", temperature, values)
        for temperature, values in grid_oof.items()]
    for decoder, key, setting, values in candidates:
        for blend in config.blend_grid:
            if blend == 0.0:
                continue  # identical to the log-GP row above
            rows.append(evaluate(
                (1 - blend) * gp_oof + blend * values,
                decoder=decoder, blend=blend, **{key: setting}))
    eligible = [row for row in rows if row["safe"]]
    # Ranked in log space, the scale on which this head models velocity: raw
    # MAPE is dominated by the slowest recipes and is far noisier over the
    # ~85 training conditions.
    selected = min(eligible, key=lambda row: (
        row["log_mae"], row["mae"], row["wape"]))
    return rows, selected, protocol, folds


# --------------------------------------------------------------------- driver
def run_grid_gp(raw_parts, config: GridGPConfig | None = None, log=print):
    """Fit both winner heads with training-OOF selection and predict all
    parts. Returns per-part condition-level predictions plus the joint GP
    latent Gaussians (for coverage) and the selection audit trail."""
    config = config or GridGPConfig()
    parts = build_bags(raw_parts)
    train_bags = parts["train"]

    log("  grid_gp: velocity decoder OOF selection")
    search_rows, selected, cv_protocol, folds = select_velocity(
        train_bags, config, log)
    log(f"  grid_gp: selected {selected['decoder']} "
        f"blend={selected['blend']} penalty={selected['penalty']} "
        f"grid_temperature={selected['grid_temperature']}")

    mapper = fit_mapper(train_bags, config, config.seed)
    mapped = {part: transform(mapper, bags) for part, bags in parts.items()}
    fit_kernel = kernel(mapped["train"], mapped["train"], mapper,
                        config.known_weight)
    joint = fit_joint(fit_kernel, train_bags, config)
    log_gp = fit_log_gp(fit_kernel, train_bags, config)
    train_velocity = np.asarray([bag["truth"][1] for bag in train_bags])
    ordinal = fit_ordinal(
        _vector(mapped["train"], config.known_weight),
        train_velocity, config, config.seed)
    grid, counts = process_grid(train_bags)
    temperature_scale = float(
        np.mean([bag["truth"][0] for bag in train_bags]))
    blend = float(selected["blend"])
    levels = np.unique(train_velocity)

    result = {"selected_velocity_decoder": selected,
              "velocity_search": search_rows,
              "velocity_inner_cv": {"protocol": cv_protocol, "folds": folds},
              "predictions": {}, "latent": {}, "latent_sigma": {},
              "conditions": {}}
    for part, bags in parts.items():
        cross = kernel(mapped[part], mapped["train"], mapper,
                       config.known_weight)
        means, covariances = predict_joint(joint, cross)
        posterior = grid_posterior(joint, means, covariances, grid, counts,
                                   config.variance_temperature)
        temperature = grid_temperature(posterior, grid, temperature_scale)
        gp_velocity = predict_log_gp(log_gp, cross)
        if selected["decoder"] == "grid_median":
            decoded = grid_velocity(
                grid_posterior(joint, means, covariances, grid, counts,
                               float(selected["grid_temperature"])), grid)
        else:
            probabilities = ordinal_probabilities(
                ordinal, _vector(mapped[part], config.known_weight))
            decoded = risk_velocity(
                ordinal[0], probabilities, gp_velocity,
                train_velocity.mean(), float(selected["penalty"]))
        velocity = snap_levels(
            (1 - blend) * gp_velocity + blend * decoded, levels)
        # The decoder acts on the variance-tempered covariance; coverage is
        # measured against that same distribution, not the raw posterior.
        tempered = (config.variance_temperature * covariances
                    + joint["noise"]
                    * np.square(joint["scale"])[None, :, None]
                    * np.eye(2)[None])
        result["predictions"][part] = np.column_stack(
            [temperature, velocity])
        result["latent"][part] = (means, covariances)
        result["latent_sigma"][part] = np.sqrt(np.maximum(
            np.stack([tempered[:, 0, 0], tempered[:, 1, 1]], axis=1), 1e-12))
        result["conditions"][part] = np.asarray(
            [bag["condition"] for bag in bags], dtype=object)
    return result


def grid_gp_report(run, part: str) -> dict:
    """Point metrics from the decoded predictions plus Gaussian coverage/NLL
    from the variance-tempered joint GP latent predictive (the distribution
    the grid decoder actually acts on)."""
    conditions = [str(c) for c in run["conditions"][part]]
    truth = LB.loc[conditions, ["T_ext", "v_ext"]].to_numpy(float)
    report = condition_level_process_metrics(
        run["predictions"][part], conditions, truth)
    means, _ = run["latent"][part]
    sigma = run["latent_sigma"][part]
    latent_truth = np.column_stack([truth[:, 0], np.log(truth[:, 1])])
    for index, target in enumerate(("T_ext", "v_ext")):
        report[target].update(gaussian_uncertainty(
            latent_truth[:, index], means[:, index], sigma[:, index],
            log_target=(target == "v_ext")))
    return report
