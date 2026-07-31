"""Manifest of conventional descriptor instances in training_data.

File naming (instance-level, per scale):
    {condition_id}_{px_scale}_{scale}_{instance}_final_2point.npy
    {condition_id}_{px_scale}_{scale}_{instance}_final_mcrpy.npy          (3-point)
    {condition_id}_{px_scale}_{scale}_{instance}_final_gram_matrices.npz
    {condition_id}_{px_scale}_{scale}_{instance}_mask_statistics_aspect_ratio.npy
    {condition_id}_{px_scale}_{scale}_{instance}_mask_statistics_eq_diameter.npy

Condition-level (texture, one per condition):
    GSH/{condition_id}_XRD.npy   (1102-dim GSH coefficient vector)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCALES = [90, 100, 120, 150]

# descriptor key -> (relative folder template, filename suffix, level)
DESCRIPTOR_SPECS: dict[str, tuple[str, str, str]] = {
    "2point": ("2point_pymks/{scale}/final", "_final_2point.npy", "instance"),
    "3point": ("3point_mcrpy/{scale}/final", "_final_mcrpy.npy", "instance"),
    "gram": ("gram/{scale}/final", "_final_gram_matrices.npz", "instance"),
    "aspect_ratio": ("aspect_ratio/{scale}", "_mask_statistics_aspect_ratio.npy", "instance"),
    "eq_diameter": ("eq_diameter/{scale}", "_mask_statistics_eq_diameter.npy", "instance"),
    "gsh": ("GSH", "_XRD.npy", "condition"),
}


@dataclass(frozen=True)
class InstanceKey:
    condition_id: str
    scale: int
    instance: int


def _parse_instance_name(name: str, scale: int, suffix: str) -> tuple[str, int] | None:
    """Return (condition_id, instance) or None if the name does not match."""
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    m = re.match(rf"^(?P<cond>.+)_(?P<px>[\d.]+)_{scale}_(?P<idx>\d+)$", stem)
    if m is None:
        return None
    return m.group("cond"), int(m.group("idx"))


def build_manifest(training_data_root: str | Path, condition_ids: list[str],
                   scales: list[int] = SCALES) -> pd.DataFrame:
    """One row per (condition, scale, instance, descriptor) with the file path.

    Only conditions present in `condition_ids` (the labels.csv universe) are kept,
    which automatically drops heat-treated conditions when they are excluded.
    """
    root = Path(training_data_root)
    keep = set(condition_ids)
    rows = []
    for desc, (folder_tpl, suffix, level) in DESCRIPTOR_SPECS.items():
        if level == "condition":
            folder = root / folder_tpl
            for f in sorted(folder.glob(f"*{suffix}")):
                cond = f.name[: -len(suffix)]
                if cond in keep:
                    rows.append({"descriptor": desc, "condition_id": cond,
                                 "scale": -1, "instance": -1, "path": str(f)})
            continue
        for scale in scales:
            folder = root / folder_tpl.format(scale=scale)
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                parsed = _parse_instance_name(f.name, scale, suffix)
                if parsed is None:
                    continue
                cond, idx = parsed
                if cond in keep:
                    rows.append({"descriptor": desc, "condition_id": cond,
                                 "scale": scale, "instance": idx, "path": str(f)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No descriptor files matched under {root}")
    return df.sort_values(["descriptor", "scale", "condition_id", "instance"]
                          ).reset_index(drop=True)
