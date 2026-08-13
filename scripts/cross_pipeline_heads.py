#!/usr/bin/env python
"""Symmetric prediction-head study across conventional, GenAI, and GNN features.

Both tasks are conditioned on the processing route: composition heads receive a
structure representation, the known extrusion parameters, and the
extrusion-ratio type; process heads receive the representation, the known alloy
composition, and the extrusion-ratio type. Each task therefore sees everything
about the other half of the condition that a deployment would already know.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

from logit_adjusted_fusion import TAUS, adjust_prior, alloy_condition_counts
from tabular_heads import (ALLOYS, ALLOY, ELEM, LB, NOMINAL, cls_targets,
                           condition_weights, labels)

PARTS = ("train", "val", "test")
RATIO_TYPES = ("standard", "mg_gd_series")
PROCESS_COLUMNS = ["T_ext", "v_ext"]
FTT_BATCH = 256
FTT_HEADS = 8
FTT_ATTENTION_BUDGET_BYTES = 1_700_000_000


def _split_conditions(split: str, part: str) -> list[str]:
    data = json.loads(Path(f"data/splits/{split}.json").read_text())
    return data[part]


def load_conventional(split: str):
    root = Path("data/conventional") / split
    return tuple(
        (np.load(root / f"X_{part}.npy"),
         np.asarray(json.loads(
             (root / f"conditions_{part}.json").read_text())))
        for part in PARTS
    )


_GENAI = None


def load_genai(split: str):
    global _GENAI
    if _GENAI is None:
        data = np.load("data/genai/latents.npz", allow_pickle=True)
        _GENAI = (
            data["latents"].reshape(len(data["latents"]), -1),
            np.asarray([str(value) for value in data["condition_ids"]]),
        )
    features, condition_ids = _GENAI
    return tuple(
        (features[np.isin(condition_ids, _split_conditions(split, part))],
         condition_ids[np.isin(
             condition_ids, _split_conditions(split, part))])
        for part in PARTS
    )


def _extract_gnn_part(split: str, part: str):
    import torch
    import yaml
    from torch_geometric.loader import DataLoader

    from icme_mg.models.gnn_encoder import GrainGraphEncoder
    from icme_mg.training.datasets import GrainGraphDataset

    flow_config = yaml.safe_load(Path("configs/flow_head.yaml").read_text())
    gnn_config = yaml.safe_load(Path("configs/gnn.yaml").read_text())[
        "gnn_encoder"]
    encoder = GrainGraphEncoder(
        hidden_dim=gnn_config["hidden_dim"],
        num_layers=gnn_config["num_layers"], heads=gnn_config["heads"],
        pool_tokens=gnn_config["pool_tokens"],
        out_dim=flow_config["head"]["dim"], dropout=gnn_config["dropout"])
    checkpoint = torch.load(
        f"runs/gnn_{split}_s90/best.pt", map_location="cuda",
        weights_only=False)
    state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(state)
    encoder.to("cuda").eval()
    dataset = GrainGraphDataset(
        "data/gnn/synthetic/graphs", "data/labels/labels.csv",
        _split_conditions(split, part), scales=[90])
    features = []
    condition_ids = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64):
            tokens = encoder(batch.to("cuda"))
            features.append(tokens.mean(dim=1).cpu().numpy())
            condition_ids.extend(batch.condition_id)
    return np.concatenate(features), np.asarray(condition_ids)


def load_gnn(split: str):
    cache_path = Path(f"data/gnn/embeddings/{split}.npz")
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        arrays = {key: cache[key] for key in cache.files}
    else:
        arrays = {}
    keys = {
        "train": ("Xtr", "ctr"),
        "val": ("Xva", "cva"),
        "test": ("Xte", "cte"),
    }
    changed = False
    for part, (feature_key, condition_key) in keys.items():
        if feature_key not in arrays or condition_key not in arrays:
            arrays[feature_key], arrays[condition_key] = _extract_gnn_part(
                split, part)
            changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **arrays)
    return (
        (arrays["Xtr"], arrays["ctr"]),
        (arrays["Xva"], arrays["cva"]),
        (arrays["Xte"], arrays["cte"]),
    )


LOADERS = {
    "conventional": load_conventional,
    "genai": load_genai,
    "gnn": load_gnn,
}


def standardize(parts):
    train = parts[0][0]
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return tuple(((features - mean) / std, conditions)
                 for features, conditions in parts)


def ratio_features(condition_ids: np.ndarray) -> np.ndarray:
    ratio_types = labels.loc[
        condition_ids, "extrusion_ratio_type"].to_numpy(str)
    return np.stack([
        ratio_types == ratio_type for ratio_type in RATIO_TYPES
    ], axis=1).astype(float)


def conditioned_inputs(parts):
    standardized = standardize(parts)
    train_conditions = sorted(set(standardized[0][1]))
    train_composition = LB.loc[train_conditions, ELEM].to_numpy(float)
    composition_mean = train_composition.mean(axis=0)
    composition_std = train_composition.std(axis=0) + 1e-8
    train_process = process_targets(train_conditions)
    process_mean = train_process.mean(axis=0)
    process_std = train_process.std(axis=0) + 1e-8
    composition_parts = []
    process_parts = []
    for representation, condition_ids in standardized:
        ratio = ratio_features(condition_ids)
        known_composition = LB.loc[
            condition_ids, ELEM].to_numpy(float)
        known_composition = (
            known_composition - composition_mean) / composition_std
        known_process = (
            process_targets(condition_ids) - process_mean) / process_std
        composition_parts.append((
            np.concatenate(
                [representation, known_process, ratio], axis=1),
            condition_ids))
        process_parts.append((
            np.concatenate(
                [representation, known_composition, ratio], axis=1),
            condition_ids))
    return tuple(composition_parts), tuple(process_parts)


def representation_coverage(split: str, parts) -> dict:
    coverage = {}
    for part, (_, condition_ids) in zip(PARTS, parts):
        expected = set(_split_conditions(split, part))
        observed = set(condition_ids)
        coverage[part] = {
            "expected_conditions": len(expected),
            "observed_conditions": len(observed),
            "missing_conditions": sorted(expected - observed),
        }
    return coverage


def condition_pool(values: np.ndarray, condition_ids: np.ndarray):
    conditions = sorted(set(condition_ids))
    pooled = np.stack([
        values[condition_ids == condition].mean(axis=0)
        for condition in conditions
    ])
    return conditions, pooled


def composition_metrics(probabilities: np.ndarray,
                        condition_ids: np.ndarray):
    conditions, pooled = condition_pool(probabilities, condition_ids)
    order = np.argsort(-pooled, axis=1)
    prediction = NOMINAL[order[:, 0]]
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    per_element = {}
    wapes = []
    for index, element in enumerate(ELEM):
        present = truth[:, index] > 0
        wape = float(
            np.abs(truth[present, index] - prediction[present, index]).sum()
            / truth[present, index].sum())
        per_element[element] = {
            "mae": float(np.abs(
                truth[:, index] - prediction[:, index]).mean()),
            "wape_present": wape,
        }
        wapes.append(wape)
    targets = cls_targets(conditions)
    return {
        "element_wape_present": float(np.mean(wapes)),
        "element_mae_macro": float(np.abs(truth - prediction).mean()),
        "alloy_top1": float(np.mean(order[:, 0] == targets)),
        "alloy_top3": float(np.mean([
            targets[row] in order[row, :3]
            for row in range(len(targets))])),
        "class_nll": float(-np.log(np.clip(
            pooled[np.arange(len(targets)), targets], 1e-12, 1)).mean()),
        "per_element": per_element,
    }


def regression_composition_metrics(predictions: np.ndarray,
                                   condition_ids: np.ndarray):
    conditions, prediction = condition_pool(predictions, condition_ids)
    truth = np.stack([LB.loc[c, ELEM].to_numpy(float) for c in conditions])
    per_element = {}
    wapes = []
    for index, element in enumerate(ELEM):
        present = truth[:, index] > 0
        wape = float(
            np.abs(truth[present, index] - prediction[present, index]).sum()
            / truth[present, index].sum())
        per_element[element] = {
            "mae": float(np.abs(
                truth[:, index] - prediction[:, index]).mean()),
            "wape_present": wape,
        }
        wapes.append(wape)
    return {
        "element_wape_present": float(np.mean(wapes)),
        "element_mae_macro": float(np.abs(truth - prediction).mean()),
        "per_element": per_element,
    }


def process_metrics(predictions: np.ndarray, condition_ids: np.ndarray):
    conditions, prediction = condition_pool(predictions, condition_ids)
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
            "r2": float(1 - np.square(residual).sum() / ss_total),
        }
    return output


def full_probabilities(model, features):
    output = np.zeros((len(features), len(ALLOYS)))
    output[:, [int(value) for value in model.classes_]] = np.asarray(
        model.predict_proba(features))
    return output


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


def fit_xgb_classifier(X_train, train_conditions,
                       n_estimators=300, max_depth=6, learning_rate=0.1):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist", device="cuda", n_jobs=-1, verbosity=0,
        random_state=0)
    model.fit(X_train, cls_targets(train_conditions))
    return model


def transform_process(values: np.ndarray) -> np.ndarray:
    """v_ext spans 1.2 decades on a near-geometric grid, so it is learned in
    log space; T_ext spans 0.4 decades and is left linear."""
    return np.column_stack([values[:, 0], np.log(values[:, 1])])


def invert_process(values: np.ndarray) -> np.ndarray:
    return np.column_stack([values[:, 0], np.exp(values[:, 1])])


def process_targets(conditions) -> np.ndarray:
    return transform_process(
        LB.loc[conditions, PROCESS_COLUMNS].to_numpy(float))


def knn_outputs(X_train, X_query, train_conditions, balanced=False,
                k=5, temperature=0.05):
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


def fit_xgb_regressors(X_train, train_conditions, columns,
                       balanced=False, targets=None,
                       n_estimators=300, max_depth=6, learning_rate=0.1):
    from xgboost import XGBRegressor

    if targets is None:
        targets = LB.loc[train_conditions, columns].to_numpy(float)
    weights = condition_weights(train_conditions) if balanced else None
    models = []
    for index in range(len(columns)):
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist", device="cuda", n_jobs=-1, verbosity=0,
            random_state=0)
        model.fit(X_train, targets[:, index], sample_weight=weights)
        models.append(model)
    return models


def predict_regressors(models, features):
    return np.stack([model.predict(features) for model in models], axis=1)


def fit_catboost_regressor(X_train, train_conditions, columns,
                           balanced=False,
                           iterations=500, depth=6, learning_rate=0.1):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="MultiRMSE", iterations=iterations, depth=depth,
        learning_rate=learning_rate, random_seed=0, verbose=0,
        boosting_type="Plain", task_type="GPU", devices="0")
    weights = condition_weights(train_conditions) if balanced else None
    model.fit(X_train, LB.loc[train_conditions, columns].to_numpy(float),
              sample_weight=weights)
    return model


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


def predict_catboost_process(fitted, features):
    model, mean, std = fitted
    return invert_process(np.asarray(model.predict(features)) * std + mean)


def _make_ftt(input_dim, output_dim):
    from rtdl_revisiting_models import FTTransformer

    return FTTransformer(
        n_cont_features=input_dim, cat_cardinalities=[],
        n_blocks=2, d_block=128, attention_n_heads=FTT_HEADS,
        attention_dropout=0.2, ffn_d_hidden_multiplier=2.0,
        ffn_dropout=0.1, residual_dropout=0.0,
        d_out=output_dim).to("cuda")


def ftt_micro_batch(n_features: int) -> int:
    """Largest chunk whose attention logits stay inside the memory budget.

    FT-Transformer emits one token per feature, so attention cost grows with
    the square of the representation width.
    """
    tokens = n_features + 1
    per_sample = FTT_HEADS * tokens * tokens * 4
    return max(1, min(FTT_BATCH, int(FTT_ATTENTION_BUDGET_BYTES
                                     // per_sample)))


def fit_ftt_classifier(X_train, train_conditions, balanced=False):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device="cuda")
    classes = torch.as_tensor(
        cls_targets(train_conditions), dtype=torch.long, device="cuda")
    # Base heads train unweighted; condition balancing lives in the balanced
    # fusion variants only, so base-head comparisons are architecture-only.
    weights = torch.as_tensor(
        condition_weights(train_conditions) if balanced
        else np.ones(len(train_conditions)),
        dtype=torch.float32, device="cuda")
    model = _make_ftt(X_train.shape[1], len(ALLOYS))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)
    micro = ftt_micro_batch(X_train.shape[1])
    for _ in range(30):
        permutation = torch.randperm(len(X_tensor), device="cuda")
        for start in range(0, len(X_tensor), FTT_BATCH):
            batch = permutation[start:start + FTT_BATCH]
            optimizer.zero_grad()
            for offset in range(0, len(batch), micro):
                indices = batch[offset:offset + micro]
                output = model(X_tensor[indices], None)
                per_sample = functional.cross_entropy(
                    output, classes[indices], reduction="none")
                loss = (per_sample * weights[indices]).sum() / len(batch)
                loss.backward()
            optimizer.step()
    return model.eval()


def predict_ftt_classifier(model, features):
    import torch
    import torch.nn.functional as functional

    tensor = torch.as_tensor(features, dtype=torch.float32, device="cuda")
    micro = ftt_micro_batch(features.shape[1])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensor), micro):
            outputs.append(model(tensor[start:start + micro], None))
    output = torch.cat(outputs)
    return functional.softmax(output, dim=1).cpu().numpy()


def fit_ftt_process(X_train, train_conditions):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device="cuda")
    process = torch.as_tensor(
        process_targets(train_conditions),
        dtype=torch.float32, device="cuda")
    train_conditions_unique = sorted(set(train_conditions))
    condition_targets = process_targets(train_conditions_unique)
    process_mean = torch.as_tensor(
        condition_targets.mean(axis=0), dtype=torch.float32, device="cuda")
    process_std = torch.as_tensor(
        condition_targets.std(axis=0) + 1e-8,
        dtype=torch.float32, device="cuda")
    process_normalized = (process - process_mean) / process_std
    model = _make_ftt(X_train.shape[1], 2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)
    micro = ftt_micro_batch(X_train.shape[1])
    for _ in range(30):
        permutation = torch.randperm(len(X_tensor), device="cuda")
        for start in range(0, len(X_tensor), FTT_BATCH):
            batch = permutation[start:start + FTT_BATCH]
            optimizer.zero_grad()
            for offset in range(0, len(batch), micro):
                indices = batch[offset:offset + micro]
                output = model(X_tensor[indices], None)
                per_sample = functional.mse_loss(
                    output, process_normalized[indices],
                    reduction="none").mean(1)
                loss = per_sample.sum() / len(batch)
                loss.backward()
            optimizer.step()
    return model.eval(), process_mean, process_std


def predict_ftt_process(fitted, features):
    import torch

    model, process_mean, process_std = fitted
    tensor = torch.as_tensor(features, dtype=torch.float32, device="cuda")
    micro = ftt_micro_batch(features.shape[1])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensor), micro):
            outputs.append(model(tensor[start:start + micro], None))
    return invert_process(
        (torch.cat(outputs) * process_std + process_mean).cpu().numpy())


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
    mean, std, pca, models, target_stats = fitted
    conditions, pooled = condition_pool(X_query, query_conditions)
    reduced = pca.transform((pooled - mean) / std)
    prediction = np.zeros((len(conditions), 2))
    sigma = np.zeros((len(conditions), 2))
    uncertainty = {}
    truth = process_targets(conditions)
    for index, target in enumerate(PROCESS_COLUMNS):
        normalized_mean, normalized_std = models[index].predict(
            reduced, return_std=True)
        target_mean, target_std = target_stats[index]
        prediction[:, index] = normalized_mean * target_std + target_mean
        predictive_std = np.maximum(normalized_std * target_std, 1e-6)
        sigma[:, index] = predictive_std
        z_score = (truth[:, index] - prediction[:, index]) / predictive_std
        # v_ext is modelled as lognormal, so its NLL needs the log Jacobian to
        # stay a density in mm/s.
        jacobian = truth[:, index].mean() if target == "v_ext" else 0.0
        uncertainty[target] = {
            "nll": float(np.mean(
                0.5 * np.square(z_score) + np.log(predictive_std)
                + 0.5 * np.log(2 * np.pi)) + jacobian),
            "coverage90": float(np.mean(np.abs(z_score) < 1.6449)),
        }
    raw_prediction = invert_process(prediction)
    median_prediction = raw_prediction.copy()
    # Smearing correction: exp(mu) is the lognormal median; the mean is
    # exp(mu + sigma^2/2). Point metrics should compare against the mean.
    raw_prediction[:, 1] = np.exp(prediction[:, 1]
                                  + 0.5 * np.square(sigma[:, 1]))
    return conditions, raw_prediction, sigma, uncertainty, median_prediction


def condition_level_process_metrics(prediction, conditions):
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
            "r2": float(1 - np.square(residual).sum() / ss_total),
        }
    return output


XGB_GRID = (
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.05},
)
CAT_GRID = (
    {"iterations": 500, "depth": 6, "learning_rate": 0.1},
    {"iterations": 900, "depth": 4, "learning_rate": 0.05},
    {"iterations": 500, "depth": 8, "learning_rate": 0.05},
)
KNN_GRID = tuple(
    (k, temperature)
    for k in (1, 3, 5, 7)
    for temperature in (0.01, 0.05, 0.1, 0.2))


def _process_score(prediction_raw, conditions):
    metrics = process_metrics(prediction_raw, conditions)
    return metrics["v_ext"]["mape"] + metrics["T_ext"]["wape"]


def tune_xgb_classifier(X_train, train_conditions, X_val, val_conditions):
    best = None
    for params in XGB_GRID:
        model = fit_xgb_classifier(X_train, train_conditions, **params)
        score = composition_metrics(full_probabilities(model, X_val),
                                    val_conditions)["element_wape_present"]
        if best is None or score < best[0]:
            best = (score, params, model)
    return best[2], dict(best[1])


def tune_catboost_classifier(X_train, train_conditions, X_val, val_conditions):
    best = None
    for params in CAT_GRID:
        model = fit_catboost(X_train, train_conditions, **params)
        score = composition_metrics(full_probabilities(model, X_val),
                                    val_conditions)["element_wape_present"]
        if best is None or score < best[0]:
            best = (score, params, model)
    return best[2], dict(best[1])


def tune_xgb_regressors(X_train, train_conditions, X_val, val_conditions,
                        columns, targets=None):
    best = None
    for params in XGB_GRID:
        models = fit_xgb_regressors(
            X_train, train_conditions, columns, targets=targets, **params)
        prediction = predict_regressors(models, X_val)
        if targets is None:
            score = regression_composition_metrics(
                prediction, val_conditions)["element_wape_present"]
        else:
            score = _process_score(invert_process(prediction), val_conditions)
        if best is None or score < best[0]:
            best = (score, params, models)
    return best[2], dict(best[1])


def tune_catboost_regressor(X_train, train_conditions, X_val, val_conditions,
                            columns):
    best = None
    for params in CAT_GRID:
        model = fit_catboost_regressor(
            X_train, train_conditions, columns, **params)
        score = regression_composition_metrics(
            np.asarray(model.predict(X_val)),
            val_conditions)["element_wape_present"]
        if best is None or score < best[0]:
            best = (score, params, model)
    return best[2], dict(best[1])


def tune_catboost_process(X_train, train_conditions, X_val, val_conditions):
    best = None
    for params in CAT_GRID:
        fitted = fit_catboost_process(X_train, train_conditions, **params)
        score = _process_score(
            predict_catboost_process(fitted, X_val), val_conditions)
        if best is None or score < best[0]:
            best = (score, params, fitted)
    return best[2], dict(best[1])


def tune_knn(X_train, train_conditions, X_val, val_conditions, task):
    best = None
    for k, temperature in KNN_GRID:
        proba, process_prediction = knn_outputs(
            X_train, X_val, train_conditions, k=k, temperature=temperature)
        if task == "composition":
            score = composition_metrics(
                proba, val_conditions)["element_wape_present"]
        else:
            score = _process_score(
                invert_process(process_prediction), val_conditions)
        if best is None or score < best[0]:
            best = (score, (k, temperature))
    return best[1]


def pca_compress_parts(parts, variance=0.95, cap=128):
    """Reduce a wide representation for the FT-Transformer, which pays one
    attention token per feature; fitted on train only."""
    from sklearn.decomposition import PCA

    train = parts[0][0]
    probe = PCA(n_components=min(cap, train.shape[0] - 1, train.shape[1]),
                random_state=0).fit(train)
    n = int(np.searchsorted(
        np.cumsum(probe.explained_variance_ratio_), variance) + 1)
    pca = PCA(n_components=n, random_state=0).fit(train)
    return tuple((pca.transform(x), c) for x, c in parts), n


def run_split(pipeline: str, split: str):
    raw_parts = LOADERS[pipeline](split)
    coverage = representation_coverage(split, raw_parts)
    if coverage["val"]["missing_conditions"] \
            or coverage["test"]["missing_conditions"]:
        raise RuntimeError(
            f"{pipeline}/{split} has incomplete validation/test coverage: "
            f"{coverage}")
    composition_parts, process_parts = conditioned_inputs(raw_parts)
    (X_train, train_conditions), (X_val, val_conditions), (
        X_test, test_conditions) = composition_parts
    (X_train_process, _), (X_val_process, _), (X_test_process, _) = (
        process_parts)

    # FT-Transformer pays one attention token per feature, so the 1284-wide
    # genai input forces micro-batches of 32; compress it (FTT only) to the
    # 95% variance dimension so it trains at the full batch like the others.
    ftt_comp_parts, ftt_proc_parts = composition_parts, process_parts
    ftt_components = None
    if pipeline == "genai":
        ftt_comp_parts, n_comp = pca_compress_parts(composition_parts)
        ftt_proc_parts, n_proc = pca_compress_parts(process_parts)
        ftt_components = {"composition": n_comp, "process": n_proc}
    X_train_ftt, X_test_ftt = ftt_comp_parts[0][0], ftt_comp_parts[2][0]
    X_train_ftt_proc, X_test_ftt_proc = (
        ftt_proc_parts[0][0], ftt_proc_parts[2][0])

    xgb_classifier, sel_xgb_cls = tune_xgb_classifier(
        X_train, train_conditions, X_val, val_conditions)
    cat_classifier, sel_cat_cls = tune_catboost_classifier(
        X_train, train_conditions, X_val, val_conditions)
    balanced_classifier = fit_catboost(
        X_train, train_conditions, balanced=True, **sel_cat_cls)
    cat_element_regressor, sel_cat_reg = tune_catboost_regressor(
        X_train, train_conditions, X_val, val_conditions, ELEM)
    cat_process_regressor, sel_cat_proc = tune_catboost_process(
        X_train_process, train_conditions, X_val_process, val_conditions)
    xgb_element_regressors, sel_xgb_reg = tune_xgb_regressors(
        X_train, train_conditions, X_val, val_conditions, ELEM)
    xgb_process_regressors, sel_xgb_proc = tune_xgb_regressors(
        X_train_process, train_conditions, X_val_process, val_conditions,
        PROCESS_COLUMNS, targets=process_targets(train_conditions))
    knn_k, knn_temp = tune_knn(
        X_train, train_conditions, X_val, val_conditions, "composition")
    knn_kp, knn_tempp = tune_knn(
        X_train_process, train_conditions, X_val_process, val_conditions,
        "process")
    ftt_classifier = fit_ftt_classifier(X_train_ftt, train_conditions)
    ftt_process = fit_ftt_process(X_train_ftt_proc, train_conditions)
    gp = fit_gp_process(X_train_process, train_conditions)

    test_knn, _ = knn_outputs(
        X_train, X_test, train_conditions, k=knn_k, temperature=knn_temp)
    _, test_knn_process = knn_outputs(
        X_train_process, X_test_process, train_conditions,
        k=knn_kp, temperature=knn_tempp)
    test_knn_process = invert_process(test_knn_process)
    val_balanced_knn, _ = knn_outputs(
        X_train, X_val, train_conditions, balanced=True,
        k=knn_k, temperature=knn_temp)
    test_balanced_knn, _ = knn_outputs(
        X_train, X_test, train_conditions, balanced=True,
        k=knn_k, temperature=knn_temp)
    test_cat = full_probabilities(cat_classifier, X_test)
    val_balanced_cat = full_probabilities(balanced_classifier, X_val)
    test_balanced_cat = full_probabilities(balanced_classifier, X_test)
    val_balanced_fuse = 0.5 * val_balanced_cat + 0.5 * val_balanced_knn
    test_balanced_fuse = 0.5 * test_balanced_cat + 0.5 * test_balanced_knn
    counts = alloy_condition_counts(train_conditions)
    tau_reports = {
        tau: composition_metrics(
            adjust_prior(val_balanced_fuse, counts, tau), val_conditions)
        for tau in TAUS
    }
    selected_tau = min(
        TAUS,
        key=lambda tau: (
            tau_reports[tau]["element_wape_present"],
            tau_reports[tau]["element_mae_macro"]))
    test_ftt = predict_ftt_classifier(ftt_classifier, X_test_ftt)
    test_ftt_process = predict_ftt_process(ftt_process, X_test_ftt_proc)

    cat_element_regression = np.asarray(
        cat_element_regressor.predict(X_test))
    cat_process_regression = predict_catboost_process(
        cat_process_regressor, X_test_process)
    test_xgb = full_probabilities(xgb_classifier, X_test)
    test_cat_knn_fusion = 0.5 * test_cat + 0.5 * test_knn
    test_prior_adjusted = adjust_prior(
        test_balanced_fuse, counts, selected_tau)
    xgb_element_regression = predict_regressors(
        xgb_element_regressors, X_test)
    xgb_process_regression = invert_process(predict_regressors(
        xgb_process_regressors, X_test_process))
    composition = {
        "knn": composition_metrics(test_knn, test_conditions),
        "xgb_classifier": composition_metrics(test_xgb, test_conditions),
        "catboost_classifier": composition_metrics(
            test_cat, test_conditions),
        "ft_transformer": composition_metrics(test_ftt, test_conditions),
        "catboost_knn_fusion": composition_metrics(
            test_cat_knn_fusion, test_conditions),
        "balanced_fusion": composition_metrics(
            test_balanced_fuse, test_conditions),
        "prior_adjusted_balanced_fusion": composition_metrics(
            test_prior_adjusted, test_conditions),
        "xgb_regression": regression_composition_metrics(
            xgb_element_regression, test_conditions),
        "catboost_regression": regression_composition_metrics(
            cat_element_regression, test_conditions),
    }

    gp_conditions, gp_prediction, gp_sigma, gp_uncertainty, gp_median = \
        predict_gp_process(gp, X_test_process, test_conditions)
    gp_report = condition_level_process_metrics(gp_prediction, gp_conditions)
    for target in gp_report:
        gp_report[target].update(gp_uncertainty[target])
    process = {
        "knn": process_metrics(test_knn_process, test_conditions),
        "xgb": process_metrics(xgb_process_regression, test_conditions),
        "catboost": process_metrics(
            cat_process_regression, test_conditions),
        "ft_transformer": process_metrics(
            test_ftt_process, test_conditions),
        "gaussian_process": gp_report,
    }
    predictions = {
        "image_condition_ids": np.asarray(test_conditions, dtype=object),
        "composition/knn/proba": test_knn,
        "composition/xgb_classifier/proba": test_xgb,
        "composition/catboost_classifier/proba": test_cat,
        "composition/ft_transformer/proba": test_ftt,
        "composition/catboost_knn_fusion/proba": test_cat_knn_fusion,
        "composition/balanced_fusion/proba": test_balanced_fuse,
        "composition/prior_adjusted_balanced_fusion/proba":
            test_prior_adjusted,
        "composition/xgb_regression/elements": xgb_element_regression,
        "composition/catboost_regression/elements": cat_element_regression,
        "process/knn/values": test_knn_process,
        "process/xgb/values": xgb_process_regression,
        "process/catboost/values": cat_process_regression,
        "process/ft_transformer/values": test_ftt_process,
        "process/gaussian_process/condition_ids":
            np.asarray(gp_conditions, dtype=object),
        "process/gaussian_process/mean": gp_prediction,
        "process/gaussian_process/median": gp_median,
        "process/gaussian_process/sigma": gp_sigma,
    }
    report = {
        "split": split,
        "selected_tau": selected_tau,
        "selected_hyperparameters": {
            "xgb_classifier": sel_xgb_cls,
            "catboost_classifier": sel_cat_cls,
            "catboost_regression": sel_cat_reg,
            "catboost_process": sel_cat_proc,
            "xgb_regression": sel_xgb_reg,
            "xgb_process": sel_xgb_proc,
            "knn_composition": {"k": knn_k, "temperature": knn_temp},
            "knn_process": {"k": knn_kp, "temperature": knn_tempp},
            "ftt_pca_components": ftt_components,
        },
        "input_dimensions": {
            "composition": int(X_train.shape[1]),
            "process": int(X_train_process.shape[1]),
        },
        "representation_coverage": coverage,
        "composition": composition,
        "process": process,
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=list(LOADERS))
    parser.add_argument("--split", default="random_seed0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()
    report, predictions = run_split(args.pipeline, args.split)
    winner = report["composition"]["prior_adjusted_balanced_fusion"]
    print(
        f"{args.pipeline}/{args.split}: tau={report['selected_tau']:.2f} "
        f"WAPE={winner['element_wape_present']:.3f}", flush=True)
    summary = {
        "protocol": "alloy_stratified_random_condition_split",
        "split": args.split,
        "pipeline": args.pipeline,
        "input_contract": {
            "composition": (
                "representation + known_T_ext_and_log_v_ext + "
                "extrusion_ratio_type_one_hot"),
            "process": (
                "representation + known_element_wt_percent + "
                "extrusion_ratio_type_one_hot"),
            "extrusion_ratio_types": list(RATIO_TYPES),
        },
        "composition_heads": report["composition"],
        "process_heads": report["process"],
        "result": report,
    }
    output = Path(args.out or (
        f"runs/summary_cross_pipeline_heads_{args.pipeline}_{args.split}.json"))
    output.write_text(json.dumps(summary, indent=2))
    prediction_path = Path(args.predictions or (
        f"runs/predictions_cross_pipeline_{args.pipeline}_{args.split}.npz"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, **predictions)
    print(json.dumps({
        "composition_heads": summary["composition_heads"],
        "process_heads": summary["process_heads"],
    }, indent=2))
    print(f"wrote {output}")
    print(f"wrote {prediction_path}")


if __name__ == "__main__":
    main()