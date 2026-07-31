#!/usr/bin/env python
"""Build a single 70-10-20 dev split (by condition, stratified by base alloy)
for fast hyperparameter iteration. Writes data/splits/dev70.json.
Deterministic given --seed. Final metrics still come from LOCO/LOAO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=float, nargs=3,
                    default=[0.7, 0.1, 0.2])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")
    rng = np.random.default_rng(args.seed)

    train, val, test = [], [], []
    for _, grp in labels.groupby("base_alloy"):
        ids = list(grp["condition_id"])
        rng.shuffle(ids)
        n = len(ids)
        n_te = max(1, int(round(args.fractions[2] * n))) if n >= 3 else 0
        n_va = max(1, int(round(args.fractions[1] * n))) if n >= 4 else 0
        n_te = min(n_te, n - 1)
        n_va = min(n_va, n - n_te - 1)
        test.extend(ids[:n_te])
        val.extend(ids[n_te:n_te + n_va])
        train.extend(ids[n_te + n_va:])

    split = {"protocol": "dev70", "seed": args.seed,
             "train": sorted(train), "val": sorted(val), "test": sorted(test)}
    assert not (set(train) & set(val) or set(train) & set(test)
                or set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(labels)

    out = data_root / "splits" / "dev70.json"
    out.write_text(json.dumps(split, indent=2))
    print(f"dev70: train={len(train)} val={len(val)} test={len(test)} "
          f"-> {out}")


if __name__ == "__main__":
    main()
