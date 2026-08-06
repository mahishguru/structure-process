"""One frozen alloy-stratified random split at the condition level."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_random_split(labels: pd.DataFrame, val_fraction: float,
                       test_fraction: float, seed: int) -> dict[str, dict]:
    """Split whole conditions once, preserving every alloy in training."""
    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for _, grp in labels.groupby("base_alloy"):
        ids = list(grp["condition_id"])
        rng.shuffle(ids)
        n_test = max(1, int(round(test_fraction * len(ids))))
        n_val = max(1, int(round(val_fraction * len(ids))))
        if n_test + n_val >= len(ids):
            raise ValueError(
                f"Not enough conditions for train/val/test in {grp.iloc[0]['base_alloy']}")
        test.extend(ids[:n_test])
        val.extend(ids[n_test:n_test + n_val])
        train.extend(ids[n_test + n_val:])
    name = f"random_seed{seed}"
    return {
        name: {
            "protocol": "alloy_stratified_random_condition_split",
            "seed": seed,
            "val_fraction_requested": val_fraction,
            "test_fraction_requested": test_fraction,
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(test),
        }
    }


def _validate(splits: dict[str, dict], labels: pd.DataFrame) -> None:
    all_ids = set(labels["condition_id"])
    for name, s in splits.items():
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert tr | va | te == all_ids, f"{name}: ids lost"
        assert not (tr & va or tr & te or va & te), f"{name}: overlap"
        train_alloys = set(labels[labels["condition_id"].isin(tr)]["base_alloy"])
        assert train_alloys == set(labels["base_alloy"]), \
            f"{name}: an alloy has no training condition"


def main(config_path: str = "configs/paths.yaml") -> Path:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_root = Path(cfg["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")

    splits = build_random_split(
        labels, cfg.get("val_fraction", 0.15),
        cfg.get("test_fraction", 0.15), cfg.get("split_seed", 0))
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
