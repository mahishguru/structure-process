"""Frozen random splits at the condition level.

Two protocols (strategy/01 section 5):
- Primary: k random folds over conditions, stratified by alloy so every alloy
    keeps at least one training condition in every fold. The historical artifact
    prefix is ``loco`` even though this is grouped random k-fold CV, not literal
    leave-one-condition-out CV.
- Optional: leave-one-alloy-out, retained only for archival analysis.

Validation conditions are carved from the training side of each fold (never from
test), also stratified by alloy. Splits are written once and shared by all
pipelines and baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stratified_condition_folds(labels: pd.DataFrame, k: int, seed: int) -> list[list[str]]:
    """Assign each condition to one of k folds, round-robin within each alloy."""
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for _, grp in labels.groupby("base_alloy"):
        ids = list(grp["condition_id"])
        rng.shuffle(ids)
        offset = int(rng.integers(k))
        for i, cid in enumerate(ids):
            folds[(i + offset) % k].append(cid)
    return folds


def _carve_val(train_ids: list[str], labels: pd.DataFrame, frac: float, seed: int
               ) -> tuple[list[str], list[str]]:
    """Split train ids into train/val, stratified by alloy, >=1 train cond/alloy."""
    rng = np.random.default_rng(seed)
    sub = labels[labels["condition_id"].isin(train_ids)]
    val: list[str] = []
    for _, grp in sub.groupby("base_alloy"):
        ids = list(grp["condition_id"])
        if len(ids) < 2:
            continue  # keep the lone condition in train
        rng.shuffle(ids)
        n_val = max(1, int(round(frac * len(ids))))
        n_val = min(n_val, len(ids) - 1)
        val.extend(ids[:n_val])
    train = [c for c in train_ids if c not in set(val)]
    return train, val


def build_loco_splits(labels: pd.DataFrame, k: int, val_fraction: float, seed: int
                      ) -> dict[str, dict]:
    """Build alloy-stratified random folds with conditions kept intact."""
    folds = _stratified_condition_folds(labels, k, seed)
    all_ids = set(labels["condition_id"])
    splits = {}
    for i, test_ids in enumerate(folds):
        train_pool = sorted(all_ids - set(test_ids))
        train, val = _carve_val(train_pool, labels, val_fraction, seed + 1000 + i)
        splits[f"loco_fold{i}"] = {
            "protocol": "random_grouped", "fold": i,
            "train": sorted(train), "val": sorted(val), "test": sorted(test_ids),
        }
    return splits


def build_loao_splits(labels: pd.DataFrame, val_fraction: float, seed: int
                      ) -> dict[str, dict]:
    splits = {}
    for j, (alloy, grp) in enumerate(sorted(labels.groupby("base_alloy"))):
        test_ids = sorted(grp["condition_id"])
        train_pool = sorted(set(labels["condition_id"]) - set(test_ids))
        train, val = _carve_val(train_pool, labels, val_fraction, seed + 2000 + j)
        splits[f"loao_{alloy}"] = {
            "protocol": "loao", "held_out_alloy": alloy,
            "train": sorted(train), "val": sorted(val), "test": test_ids,
        }
    return splits


def _validate(splits: dict[str, dict], labels: pd.DataFrame) -> None:
    all_ids = set(labels["condition_id"])
    for name, s in splits.items():
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert tr | va | te == all_ids, f"{name}: ids lost"
        assert not (tr & va or tr & te or va & te), f"{name}: overlap"
        if s["protocol"] == "random_grouped":
            train_alloys = set(labels[labels["condition_id"].isin(tr)]["base_alloy"])
            assert train_alloys == set(labels["base_alloy"]), \
                f"{name}: an alloy has no training condition"


def main(config_path: str = "configs/paths.yaml") -> Path:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_root = Path(cfg["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")

    splits = {}
    splits.update(build_loco_splits(labels, cfg.get("loco_folds", 5),
                                    cfg.get("val_fraction", 0.15),
                                    cfg.get("split_seed", 0)))
    if cfg.get("include_loao", False):
        splits.update(build_loao_splits(labels, cfg.get("val_fraction", 0.15),
                                        cfg.get("split_seed", 0)))
    _validate(splits, labels)

    out_dir = data_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(s, indent=2))
    print(f"Wrote {len(splits)} split files -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    main(args.config)
