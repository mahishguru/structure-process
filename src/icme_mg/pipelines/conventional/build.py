"""Per-image conventional descriptor dataset (multi-scale).

One sample = one experimental image (Acta-paper granularity, nothing
oversampled, no cross-scale slot pairing). The feature vector uses the Acta
best-performing combination plus the grain-statistics histograms, taken
directly from the pre-reduced arrays in the training_data directory:

    3point_mcrpy_reduced_normalized/{scale}/final/{stem}_final_mcrpy_reduced.npy
        PCA-reduced 3-point spatial correlations            (120,)
    gram_reduced_normalized/{scale}/final/{stem}_final_gram_matrices_reduced.npy
        Isomap-reduced Gram matrices                        (200,)
    aspect_ratio/{scale}/{stem}_mask_statistics_aspect_ratio.npy
        Binned aspect-ratio histogram, no reduction         (29,)
    eq_diameter/{scale}/{stem}_mask_statistics_eq_diameter.npy
        Binned equivalent-diameter histogram (5-200 um)     (29,)
    GSH_reduced/isomap/{cid}_GSH_isomap.npy
        Isomap-reduced GSH texture coefficients, condition-
        level (XRD), broadcast to every image of the cid    (60,)

    x = [3pt(120) | gram(200) | ar(29) | eqd(29) | gsh(60)] -> 438-D,
    standardized on the training fold only.

When rve_manifest_root is set, images are filtered to the stems present in
manifest_{scale}.csv so the conventional corpus matches the GenAI and GNN
pipelines exactly (same images, same N).

Caveat recorded here on purpose: the reductions themselves were fit on the
full Acta corpus, so fold isolation applies to the standardizer only. This
mirrors the Acta paper's protocol and keeps the descriptors identical to it.

Output per split: X_{part}.npy, conditions_{part}.json (per-image condition
ids, consumed by ConventionalDataset), image_stems_{part}.json,
feature_names.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BLOCKS = [("3point", 120), ("gram", 200), ("ar", 29), ("eqd", 29),
          ("gsh", 60)]

# {cid}_{um_per_px}[_extra*]_{scale}_{imgidx}  (um_per_px always < 1)
IMAGE_STEM_RE = re.compile(
    r"^(?P<cid>.+?)_(?P<um>0\.\d+)(?:_(?P<batch>extra\w*))?"
    r"_(?P<scale>\d+)_(?P<idx>\d+)$")
MASK_SUFFIX = "_mask_statistics"


def image_index(training_data_root: str | Path, condition_ids: list[str],
                scale: int = 150,
                allowed_stems: set[str] | None = None) -> pd.DataFrame:
    """One row per image with resolved descriptor paths; skips images with
    missing files. Images are enumerated from mask_stats/{scale} (the most
    accurate per-image inventory: one file per micrograph crop, no empty
    extraction artifacts). allowed_stems, when given, restricts the index to
    those image stems (parity with the RVE-based pipelines)."""
    root = Path(training_data_root)
    p3_dir = root / "3point_mcrpy_reduced_normalized" / str(scale) / "final"
    gr_dir = root / "gram_reduced_normalized" / str(scale) / "final"
    ar_dir = root / "aspect_ratio" / str(scale)
    eq_dir = root / "eq_diameter" / str(scale)
    gsh_dir = root / "GSH_reduced" / "isomap"
    cids = set(condition_ids)
    rows, skipped = [], 0
    for f in sorted((root / "mask_stats" / str(scale))
                    .glob(f"*{MASK_SUFFIX}.csv")):
        stem = f.stem[: -len(MASK_SUFFIX)]
        m = IMAGE_STEM_RE.match(stem)
        if m is None or m["cid"] not in cids:
            continue
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        cid = m["cid"]
        p3 = p3_dir / f"{stem}_final_mcrpy_reduced.npy"
        gr = gr_dir / f"{stem}_final_gram_matrices_reduced.npy"
        ar = ar_dir / f"{stem}{MASK_SUFFIX}_aspect_ratio.npy"
        eq = eq_dir / f"{stem}{MASK_SUFFIX}_eq_diameter.npy"
        gsh = gsh_dir / f"{cid}_GSH_isomap.npy"
        if not (p3.exists() and gr.exists() and ar.exists() and eq.exists()
                and gsh.exists()):
            skipped += 1
            continue
        rows.append({"condition_id": cid, "image_stem": stem,
                     "scale": scale, "3point": str(p3), "gram": str(gr),
                     "ar": str(ar), "eqd": str(eq), "gsh": str(gsh)})
    if skipped:
        print(f"  scale {scale}: skipped {skipped} images with missing "
              "descriptors")
    return pd.DataFrame(rows)


def _feature_matrix(index: pd.DataFrame) -> np.ndarray:
    cols = []
    for key, dim in BLOCKS:
        block = np.stack([np.load(p).ravel()[:dim] for p in index[key]])
        cols.append(block.astype(np.float32))
    return np.concatenate(cols, axis=1)


class ConventionalImageBuilder:
    """Builds per-image train/val/test matrices for condition-grouped splits."""

    def __init__(self, training_data_root: str | Path, labels: pd.DataFrame,
                 scales: list[int] | int = 150,
                 rve_manifest_root: str | Path | None = None):
        if isinstance(scales, int):
            scales = [scales]
        frames = []
        for scale in scales:
            allowed = None
            if rve_manifest_root is not None:
                mf = Path(rve_manifest_root) / f"manifest_{scale}.csv"
                if mf.exists():
                    allowed = set(pd.read_csv(mf)["image_stem"])
                else:
                    print(f"  no RVE manifest for scale {scale}; "
                          "using all labeled images")
            frames.append(image_index(training_data_root,
                                      list(labels["condition_id"]), scale,
                                      allowed))
        self.index = pd.concat(frames, ignore_index=True)
        if self.index.empty:
            raise RuntimeError("no images resolved; check training_data_root")
        counts = self.index.groupby("scale").size().to_dict()
        print(f"  index: {len(self.index)} images {counts}")

    def build(self, split: dict, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = {p: self.index[self.index["condition_id"].isin(split[p])]
                 for p in ("train", "val", "test")}
        if parts["train"].empty:
            raise RuntimeError("no training images resolved for this split")

        X_train = _feature_matrix(parts["train"])
        scaler = StandardScaler().fit(X_train)
        dim = X_train.shape[1]
        result = {"feature_dim": int(dim)}
        for part, idx in parts.items():
            X = _feature_matrix(idx) if len(idx) else np.zeros((0, dim),
                                                               np.float32)
            Xs = scaler.transform(X).astype(np.float32) if len(X) else X
            np.save(out_dir / f"X_{part}.npy", Xs)
            (out_dir / f"conditions_{part}.json").write_text(
                json.dumps(list(idx["condition_id"])))
            (out_dir / f"image_stems_{part}.json").write_text(
                json.dumps(list(idx["image_stem"])))
            result[f"n_{part}"] = len(idx)
        names = [f"{key}_{i}" for key, dim in BLOCKS for i in range(dim)]
        (out_dir / "feature_names.json").write_text(json.dumps(names))
        return result


def main(config_path: str = "configs/conventional.yaml") -> None:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")

    builder = ConventionalImageBuilder(
        paths["training_data_root"], labels,
        scales=cfg.get("scales", cfg.get("scale", 150)),
        rve_manifest_root=cfg.get("rve_manifest_root"))
    split_names = cfg.get("splits") or sorted(
        p.stem for p in (data_root / "splits").glob("*.json"))
    for name in split_names:
        split = json.loads((data_root / "splits" / f"{name}.json").read_text())
        try:
            out = builder.build(split, data_root / "conventional" / name)
            print(f"{name}: {out}")
        except Exception as e:  # noqa: BLE001 - keep remaining splits going
            print(f"{name}: FAILED ({e})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conventional.yaml")
    args = ap.parse_args()
    main(args.config)
