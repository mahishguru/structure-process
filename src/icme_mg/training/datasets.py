"""Datasets for the three descriptor pipelines. Every sample carries the
condition_id so labels, balanced sampling, and per-condition aggregation stay
consistent across pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from icme_mg import ELEMENTS, LABEL_ORDER


def load_labels(labels_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    return df.set_index("condition_id")


def labels_tensor(labels: pd.DataFrame, condition_ids: list[str]
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (y_physical (N,10), present (N,8) bool, class_idx (N,))."""
    sub = labels.loc[condition_ids]
    y = torch.tensor(sub[LABEL_ORDER].to_numpy(float), dtype=torch.float32)
    present = y[:, :len(ELEMENTS)] > 0
    alloys = sorted(labels["alloy"].unique())
    class_idx = torch.tensor(
        [alloys.index(a) for a in sub["alloy"]], dtype=torch.long)
    return y, present, class_idx


class ConventionalDataset(Dataset):
    """Reads X_{part}.npy + conditions_{part}.json emitted by
    ConventionalImageBuilder for one fold."""

    def __init__(self, fold_dir: str | Path, part: str, labels_csv: str | Path):
        fold_dir = Path(fold_dir)
        self.X = torch.tensor(np.load(fold_dir / f"X_{part}.npy"),
                              dtype=torch.float32)
        self.condition_ids = json.loads(
            (fold_dir / f"conditions_{part}.json").read_text())
        labels = load_labels(labels_csv)
        self.y, self.present, self.class_idx = labels_tensor(
            labels, self.condition_ids)

    def __len__(self):
        return len(self.condition_ids)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "present": self.present[i],
                "class_idx": self.class_idx[i],
                "condition_id": self.condition_ids[i]}


class GenAILatentDataset(Dataset):
    """Reads latents.npz (keys: latents (N,16,80), condition_ids, image_names)
    emitted by pipelines/genai/extract_latents.py, filtered to a condition set."""

    def __init__(self, latents_npz: str | Path, labels_csv: str | Path,
                 conditions: list[str] | None = None):
        z = np.load(latents_npz, allow_pickle=True)
        lat, cids = z["latents"], [str(c) for c in z["condition_ids"]]
        if conditions is not None:
            keep = [i for i, c in enumerate(cids) if c in set(conditions)]
            lat, cids = lat[keep], [cids[i] for i in keep]
        self.X = torch.tensor(lat, dtype=torch.float32)
        self.condition_ids = cids
        labels = load_labels(labels_csv)
        self.y, self.present, self.class_idx = labels_tensor(labels, cids)

    def __len__(self):
        return len(self.condition_ids)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "present": self.present[i],
                "class_idx": self.class_idx[i],
                "condition_id": self.condition_ids[i]}


class GrainGraphDataset(Dataset):
    """Reads graphs saved by pipelines/gnn/build.py (data/gnn/{source}/graphs).
    Collate with torch_geometric.loader.DataLoader. `scales` optionally
    restricts to image scales (stem suffix ..._{scale}_{idx})."""

    def __init__(self, graph_dir: str | Path, labels_csv: str | Path,
                 conditions: list[str] | None = None,
                 scales: list[int] | None = None):
        graph_dir = Path(graph_dir)
        paths = sorted(graph_dir.glob("*.pt"))
        if conditions is not None:
            cset = set(conditions)
            paths = [p for p in paths if p.stem.split("__")[0] in cset]
        if scales is not None:
            sset = {str(s) for s in scales}
            paths = [p for p in paths if p.stem.split("_")[-2] in sset]
        self.paths = paths
        self.condition_ids = [p.stem.split("__")[0] for p in paths]
        labels = load_labels(labels_csv)
        self.y, self.present, self.class_idx = labels_tensor(
            labels, self.condition_ids)
        self._cache: dict[int, object] = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if i in self._cache:
            g = self._cache[i]
        else:
            g = torch.load(self.paths[i], weights_only=False)
            g.y = self.y[i].unsqueeze(0)
            g.present = self.present[i].unsqueeze(0)
            g.class_idx = self.class_idx[i].unsqueeze(0)
            g.condition_id = self.condition_ids[i]
            self._cache[i] = g
        return g


class BalancedConditionSampler(torch.utils.data.Sampler):
    """Per-condition-balanced sampling: every condition contributes equally per
    epoch regardless of instance count (strategy/01 section 5)."""

    def __init__(self, condition_ids: list[str], samples_per_condition: int = 8,
                 seed: int = 0):
        self.groups: dict[str, list[int]] = {}
        for i, c in enumerate(condition_ids):
            self.groups.setdefault(c, []).append(i)
        self.spc = samples_per_condition
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.groups) * self.spc

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = []
        for members in self.groups.values():
            pick = torch.randint(len(members), (self.spc,), generator=g)
            idx.extend(members[p] for p in pick.tolist())
        order = torch.randperm(len(idx), generator=g).tolist()
        yield from (idx[o] for o in order)
