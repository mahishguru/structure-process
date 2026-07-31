"""Render DREAM.3D RVEs to orientation-codec RGB images for the GenAI encoder.

The ViT-FMDiT-1280 encoder was trained on orientation-codec encoded images
(stereographic projection of grain quaternions centered on the class-mean
quaternion, Co-PiLOT / NeurIPS 2026 pipeline). Experimental grayscale
micrographs are out-of-distribution for it. So the GenAI pipeline consumes the
same synthetic RVEs as the GNN pipeline:

    data/gnn/rves/{condition_id}/rve/{cid}__{um}_{scale}_{imgidx}.dream3d
        --orientation_codec.encode_dream3d_global-->
    data/genai/images/{cid}__{um}_{scale}_{imgidx}.png   (8-bit RGB)

One RVE (and hence one PNG) per experimental image; PNG names reuse the
.dream3d stem so every sample stays traceable to its source micrograph.

Class-mean quaternions come from class_means.json (keys are alloy-class names
like "AZ31_extruded"); condition ids are matched by longest key prefix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def class_key_for_condition(condition_id: str, means: dict) -> str:
    """Longest class_means key that prefixes the condition id."""
    best = None
    for key in means:
        if condition_id.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is None:
        raise KeyError(f"no class_means key matches {condition_id!r}")
    return best


def render_condition(condition_id: str, rve_dir: Path, out_dir: Path,
                     means: dict) -> list[Path]:
    """Encode every .dream3d under rve_dir to a PNG named by its file stem.

    Resumable: RVEs whose PNG already exists are skipped.
    """
    from orientation_codec.dataset import encode_dream3d_global

    mean_q = np.asarray(means[class_key_for_condition(condition_id, means)])
    files = sorted(rve_dir.rglob("*.dream3d"))
    if not files:
        raise FileNotFoundError(f"no .dream3d files under {rve_dir}")
    out = []
    for f in files:
        dst = out_dir / f"{f.stem}.png"
        if dst.exists():
            out.append(dst)
            continue
        out.append(encode_dream3d_global(
            f, mean_q, output_dir=out_dir, name=f.stem, fmt="png"))
    return out


def main(config_path: str = "configs/genai.yaml") -> None:
    import yaml
    import pandas as pd
    from orientation_codec.dataset import load_class_means

    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    data_root = Path(paths["data_root"])
    labels = pd.read_csv(data_root / "labels" / "labels.csv")
    means = load_class_means(cfg["class_means"])

    rve_root = Path(cfg["rve_root"])
    out_dir = Path(cfg["image_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    done, failures = 0, []
    for cid in labels["condition_id"]:
        rve_dir = rve_root / cid
        if not rve_dir.exists():
            failures.append((cid, "no RVE directory"))
            continue
        try:
            imgs = render_condition(cid, rve_dir, out_dir, means)
            done += len(imgs)
            print(f"{cid}: {len(imgs)} images")
        except Exception as e:  # noqa: BLE001 - keep batch running
            failures.append((cid, str(e)))
            print(f"{cid}: FAILED ({e})")
    print(f"\nRendered {done} images -> {out_dir}")
    if failures:
        print(f"{len(failures)} conditions failed:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/genai.yaml")
    args = ap.parse_args()
    main(args.config)
