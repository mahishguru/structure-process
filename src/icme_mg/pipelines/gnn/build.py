"""Build grain-graph datasets for both GNN sources.

Synthetic:    per-condition DREAM.3D RVEs (see generate_rves.py) -> graphs.
Experimental: binarized OM masks from training_data (binary folders per scale)
              -> graphs, matched to condition_ids by filename prefix.

Output: data/gnn/{synthetic,experimental}/graphs/{condition_id}__{i}.pt
        + manifest.csv (condition_id, source, path, n_nodes, n_edges)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .graphs import graph_from_dream3d, graph_from_mask


def _save(graph, out_dir: Path, condition_id: str, name: str) -> dict:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pt"
    torch.save(graph, path)
    return {"condition_id": condition_id, "path": str(path),
            "n_nodes": graph.num_nodes, "n_edges": graph.num_edges}


def build_synthetic(rve_root: str | Path, condition_ids: list[str],
                    out_root: Path, min_grain_px: int = 5) -> pd.DataFrame:
    """rve_root/{condition_id}/**/{cid}__{suffix}.dream3d -> graphs.

    Graph files reuse the per-image .dream3d stem, keeping the "{cid}__"
    prefix so GrainGraphDataset condition matching still works.
    """
    rows = []
    for cid in condition_ids:
        cond_dir = Path(rve_root) / cid
        if not cond_dir.is_dir():
            continue
        for f in sorted(cond_dir.glob("**/*.dream3d")):
            existing = out_root / "graphs" / f"{f.stem}.pt"
            if existing.exists():
                import torch
                g = torch.load(existing, weights_only=False)
                rows.append({"condition_id": cid, "path": str(existing),
                             "n_nodes": g.num_nodes, "n_edges": g.num_edges,
                             "source": "synthetic"})
                continue
            try:
                g = graph_from_dream3d(f, min_grain_px)
            except Exception as e:  # noqa: BLE001 - skip corrupt RVEs, log them
                print(f"  skip {f.name}: {e}")
                continue
            rows.append({**_save(g, out_root / "graphs", cid, f.stem),
                         "source": "synthetic"})
    return pd.DataFrame(rows)


def build_experimental(training_data_root: str | Path, condition_ids: list[str],
                       out_root: Path, scales: list[int] | None = None,
                       min_grain_px: int = 5) -> pd.DataFrame:
    """training_data/binary/{scale}/*.npy masks -> graphs (prefix-matched)."""
    scales = scales or [90, 100, 120, 150]
    conds = sorted(condition_ids, key=len, reverse=True)
    rows = []
    counters: dict[str, int] = {}
    for scale in scales:
        folder = Path(training_data_root) / "binary" / str(scale)
        if not folder.is_dir():
            folder = Path(training_data_root) / "2point_pymks" / str(scale) / "binary"
        if not folder.is_dir():
            print(f"  no binary folder for scale {scale}")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix not in {".npy", ".png", ".tif", ".tiff"}:
                continue
            cid = next((c for c in conds if f.stem.startswith(c)), None)
            if cid is None:
                continue
            try:
                g = graph_from_mask(f, min_grain_px)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {f.name}: {e}")
                continue
            i = counters.get(cid, 0)
            counters[cid] = i + 1
            rows.append({**_save(g, out_root / "graphs", cid, f"{cid}__{i}"),
                         "source": f"experimental_{scale}"})
    return pd.DataFrame(rows)


def main(config_path: str = "configs/gnn.yaml") -> None:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")
    cids = list(labels["condition_id"])

    manifests = []
    if cfg.get("build_synthetic", True):
        out = data_root / "gnn" / "synthetic"
        df = build_synthetic(cfg["rve_root"], cids, out,
                             cfg.get("min_grain_px", 5))
        if len(df):
            df.to_csv(out / "manifest.csv", index=False)
        manifests.append(("synthetic", len(df)))
    if cfg.get("build_experimental", True):
        out = data_root / "gnn" / "experimental"
        df = build_experimental(paths["training_data_root"], cids, out,
                                cfg.get("scales"), cfg.get("min_grain_px", 5))
        if len(df):
            df.to_csv(out / "manifest.csv", index=False)
        manifests.append(("experimental", len(df)))
    for name, n in manifests:
        print(f"{name}: {n} graphs")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gnn.yaml")
    args = ap.parse_args()
    main(args.config)
