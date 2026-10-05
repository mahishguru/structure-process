"""Data loading and conditioning for the head benchmark.

One importable home for what used to live at module level in
scripts/tabular_heads.py and scripts/cross_pipeline_heads.py: the canonical
label table, the alloy lattice, split loaders for the three representations,
and the symmetric input contracts.

Task A (composition) input: representation + known (T_ext, log v_ext) +
extrusion-ratio one-hot.
Task B (process) input:     representation + known element wt% +
extrusion-ratio one-hot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from icme_mg import ELEMENTS

PARTS = ("train", "val", "test")
RATIO_TYPES = ("standard", "mg_gd_series")
PROCESS_COLUMNS = ["T_ext", "v_ext"]

ELEM = list(ELEMENTS)

labels = pd.read_csv("data/labels/labels.csv").set_index("condition_id")
LB = labels[ELEM + PROCESS_COLUMNS]
ALLOY = labels["alloy"]
ALLOYS = sorted(labels["alloy"].unique())
# Nominal composition per alloy in ELEMENTS order; the decode target of every
# composition classifier.
NOMINAL = np.stack([labels[labels["alloy"] == a].iloc[0][ELEM]
                    .to_numpy(float) for a in ALLOYS])


def cls_targets(condition_ids) -> np.ndarray:
    return np.array([ALLOYS.index(a) for a in ALLOY.loc[condition_ids]])


def condition_weights(condition_ids) -> np.ndarray:
    """Per-image weights giving every condition equal total mass."""
    condition_ids = np.asarray(condition_ids)
    _, inverse, counts = np.unique(
        condition_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def split_conditions(split: str, part: str) -> list[str]:
    data = json.loads(Path(f"data/splits/{split}.json").read_text())
    return data[part]


# --------------------------------------------------------------------- loaders
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
        (features[np.isin(condition_ids, split_conditions(split, part))],
         condition_ids[np.isin(
             condition_ids, split_conditions(split, part))])
        for part in PARTS
    )


def _extract_gnn_part(split: str, part: str):
    """Frozen-encoder embedding extraction. Only runs when the embedding cache
    is missing; the encoder checkpoint is the one trained on this split."""
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
        split_conditions(split, part), scales=[90])
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

PIPELINES = tuple(LOADERS)


# ---------------------------------------------------------------- conditioning
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


def transform_process(values: np.ndarray) -> np.ndarray:
    """v_ext spans 1.2 decades on a near-geometric grid, so it is learned in
    log space; T_ext spans 0.4 decades and is left linear."""
    return np.column_stack([values[:, 0], np.log(values[:, 1])])


def invert_process(values: np.ndarray) -> np.ndarray:
    return np.column_stack([values[:, 0], np.exp(values[:, 1])])


def process_targets(conditions) -> np.ndarray:
    return transform_process(
        LB.loc[conditions, PROCESS_COLUMNS].to_numpy(float))


def conditioned_inputs(parts):
    """Build the two symmetric task inputs from raw representation parts."""
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
        expected = set(split_conditions(split, part))
        observed = set(condition_ids)
        coverage[part] = {
            "expected_conditions": len(expected),
            "observed_conditions": len(observed),
            "missing_conditions": sorted(expected - observed),
        }
    return coverage


def condition_pool(values: np.ndarray, condition_ids: np.ndarray):
    """Average image-level rows to one row per condition."""
    conditions = sorted(set(condition_ids))
    pooled = np.stack([
        values[condition_ids == condition].mean(axis=0)
        for condition in conditions
    ])
    return conditions, pooled
