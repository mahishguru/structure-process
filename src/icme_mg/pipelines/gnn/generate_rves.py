"""Per-image DREAM.3D RVE generation from per-image mask grain statistics.

One RVE per experimental image, matching the Acta paper's per-image sample
granularity (no stochastic replicas, nothing oversampled). Inputs are the
per-image MASK grain statistics (the most accurate per-grain source: raw
regionprops of the binarized micrograph crop, no border-grain exclusion, no
empty extraction artifacts) and the per-condition experimental ODF:

    training_data/mask_stats/{scale}/{stem}_mask_statistics.csv
        columns Width, Height, Area (pixel units) -> 5 copula params fitted
        exactly like Sampling/01_extract_copula_params.py (lognormal ESD,
        beta AR on [0,1], Gaussian copula rho). No pooling fallback: images
        with fewer than 10 usable grains are skipped.
    sampling_repo/odf_harmonics/{cid}_odf_harmonics.txt
        per-condition GSH coefficients (XRD); every image of a condition
        shares its condition's texture.

Pipeline (all interfaces verified against the actual scripts):
    1. image_inventory()  scan mask_stats/{scale}, alias each image to an
                          integer sample id per material (04_generate_rves.py
                          only matches "{material}_(\\d+)_..." file names).
    2. fit_copulas()      per-image copula params -> staging/copula tree.
    3. ensure_odf_cache() one MTEX conversion per condition (cached at
                          staging/odf_cache; seeded from any previous staging
                          runs), then copy the condition's angle file to every
                          image id under staging/odf.
    4. run_stats()        03_copula_to_dream3d_stats.py.
    5. run_dream3d()      04_generate_rves.py with env overrides,
                          --local-staging staging/rves_raw --no-transfer.
    6. collect()          rve_root/{cid}/rve/{cid}__{um}_{scale}_{imgidx}
                          .dream3d + manifest.csv. The "{cid}__" prefix keeps
                          downstream split("__")[0] condition matching intact.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
MATLAB_SCRIPT = REPO_ROOT / "scripts" / "matlab" / "harmonics_to_dream3d_angles.m"

# {cid}_{um_per_px}[_extra*]_{scale}_{imgidx}  (um_per_px always < 1)
IMAGE_STEM_RE = re.compile(
    r"^(?P<cid>.+?)_(?P<um>0\.\d+)(?:_(?P<batch>extra\w*))?"
    r"_(?P<scale>\d+)_(?P<idx>\d+)$")
MASK_SUFFIX = "_mask_statistics"


# --------------------------------------------------------------- inventory
def image_inventory(labels, mask_stats_dir: Path, scale: int = 150
                    ) -> list[dict]:
    """One row per experimental image of a labelled condition.

    Row: material, condition_id, image_stem, um_per_px, image_idx, sample_id,
    stats_csv. sample_id is a sequential integer unique within each material.
    """
    cids = set(labels["condition_id"])
    cid_material = dict(zip(labels["condition_id"], labels["alloy"]))
    rows = []
    for f in sorted(mask_stats_dir.glob(f"*{MASK_SUFFIX}.csv")):
        stem = f.stem[: -len(MASK_SUFFIX)]
        m = IMAGE_STEM_RE.match(stem)
        if m is None:
            continue
        cid = m["cid"]
        if cid not in cids or int(m["scale"]) != scale:
            continue
        rows.append({"material": cid_material[cid], "condition_id": cid,
                     "image_stem": stem, "um_per_px": float(m["um"]),
                     "image_idx": int(m["idx"]), "stats_csv": str(f)})
    counters: dict[str, int] = {}
    for row in sorted(rows, key=lambda r: (r["material"], r["image_stem"])):
        sid = counters.get(row["material"], 0) + 1
        counters[row["material"]] = sid
        row["sample_id"] = sid
    return rows


# ------------------------------------------------------------ copula fitting
def _esd_ar_from_csv(stats_csv: str | Path, um_per_px: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-grain ESD (um) and AR in (0, 1) from one mask_statistics CSV.

    CSV columns are pixel units: Width, Height, Area (raw mask regionprops).
    ESD = 2*sqrt(Area/pi)*um_per_px; AR = min/max of bbox dims.
    """
    import pandas as pd

    df = pd.read_csv(stats_csv)
    areas = df["Area"].to_numpy(float)
    w = df["Width"].to_numpy(float) * um_per_px
    h = df["Height"].to_numpy(float) * um_per_px
    esd = 2.0 * np.sqrt(areas / np.pi) * um_per_px
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.minimum(w, h) / np.maximum(w, h)
    mask = np.isfinite(esd) & np.isfinite(ar) & (esd > 0) & (ar > 0) & (ar < 1)
    return esd[mask], ar[mask]


def _fit_copula(esd: np.ndarray, ar: np.ndarray
                ) -> tuple[float, float, float, float, float]:
    """Replicates Sampling/01_extract_copula_params.py: lognormal ESD,
    beta AR on [0,1], Gaussian copula rho."""
    from scipy.stats import beta as beta_dist
    from scipy.stats import norm

    if esd.size < 10:
        raise ValueError(f"only {esd.size} usable grains")
    log_esd = np.log(esd)
    mu, sigma = norm.fit(log_esd)
    alpha, beta_p, _, _ = beta_dist.fit(ar, floc=0, fscale=1)
    eps = 1e-6
    z_esd = norm.ppf(np.clip(norm.cdf(log_esd, mu, sigma), eps, 1 - eps))
    z_ar = norm.ppf(np.clip(beta_dist.cdf(ar, alpha, beta_p), eps, 1 - eps))
    rho = float(np.corrcoef(z_esd, z_ar)[0, 1])
    return float(mu), float(sigma), float(alpha), float(beta_p), rho


def fit_copula_from_csv(stats_csv: str | Path, um_per_px: float
                        ) -> tuple[float, float, float, float, float]:
    return _fit_copula(*_esd_ar_from_csv(stats_csv, um_per_px))


def fit_copulas(rows: list[dict], staging: Path) -> list[str]:
    """Fit + write staging/copula/{mat}/{mat}_{sid}.txt. Returns failed stems.

    Strictly per-image, no pooling fallback: each copula is fitted only on
    the grains of its own image's mask statistics. Images with fewer than 10
    usable grains fail the fit and are skipped downstream.
    """
    pending = [r for r in rows
               if not (staging / "copula" / r["material"]
                       / f"{r['material']}_{r['sample_id']}.txt").exists()]
    if not pending:
        return []

    failed = []
    for row in pending:
        mat, sid = row["material"], row["sample_id"]
        dst = staging / "copula" / mat / f"{mat}_{sid}.txt"
        esd, ar = _esd_ar_from_csv(row["stats_csv"], row["um_per_px"])
        try:
            params = _fit_copula(esd, ar)
        except Exception as e:  # noqa: BLE001 - skip unfittable images
            print(f"  copula fit failed for {row['image_stem']}: {e}")
            failed.append(row["image_stem"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("# mu_log_esd sigma_log_esd alpha_ar beta_ar rho\n"
                       + "".join(f"{p}\n" for p in params))
    return failed


# ------------------------------------------------------------------ ODF side
def seed_odf_cache_from(old_staging_odf: Path, labels, cache: Path) -> int:
    """Reuse angle files from the previous per-condition staging run.

    The old alias scheme was sample_id = local_condition_index*10 + replica;
    replica-1 files are exact MTEX conversions of the condition harmonics and
    remain valid for every image of that condition.
    """
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    for material, grp in labels.groupby("alloy"):
        for j, cid in enumerate(sorted(grp["condition_id"])):
            dst = cache / f"{cid}_dream3d_odf_angles.txt"
            if dst.exists():
                continue
            src = (old_staging_odf / material / "dream3d_angles_uniform" /
                   f"{material}_{j * 10 + 1}_dream3d_odf_angles.txt")
            if src.exists():
                shutil.copy(src, dst)
                n += 1
    return n


def ensure_odf_cache(conditions: list[str], sampling_repo: Path, cache: Path,
                     staging: Path, matlab_bin: str, mtex_root: str
                     ) -> list[str]:
    """Guarantee cache/{cid}_dream3d_odf_angles.txt for every condition.

    Missing conversions are run through MTEX once per condition. Returns
    conditions with no ODF at all (skipped downstream).
    """
    cache.mkdir(parents=True, exist_ok=True)
    no_odf, pending = [], []
    for cid in conditions:
        if (cache / f"{cid}_dream3d_odf_angles.txt").exists():
            continue
        src = sampling_repo / "odf_harmonics" / f"{cid}_odf_harmonics.txt"
        if not src.exists():
            no_odf.append(cid)
            continue
        work = staging / "odf_mtex"
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, work / f"{cid}.txt")
        pending.append(cid)
    if pending:
        print(f"  MTEX conversion for {len(pending)} conditions")
        env = dict(os.environ, MTEX_ROOT=str(mtex_root),
                   ODF_PARENT_DIR=str((staging / "odf_mtex").resolve()))
        cmd = [matlab_bin, "-batch", f"run('{MATLAB_SCRIPT}')"]
        print("  $", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=env)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  MTEX step failed ({e}); continuing with cache")
        for cid in pending:
            out = (staging / "odf_mtex" / "dream3d_angles_uniform" /
                   f"{cid}_dream3d_odf_angles.txt")
            if out.exists():
                shutil.copy(out, cache / f"{cid}_dream3d_odf_angles.txt")
            else:
                no_odf.append(cid)
    return sorted(set(no_odf))


def stage_angle_files(rows: list[dict], cache: Path, staging: Path) -> None:
    """Copy each condition's cached angle file to every image sample id."""
    for row in rows:
        mat, sid, cid = row["material"], row["sample_id"], row["condition_id"]
        src = cache / f"{cid}_dream3d_odf_angles.txt"
        if not src.exists():
            continue
        dst_dir = staging / "odf" / mat / "dream3d_angles_uniform"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{mat}_{sid}_dream3d_odf_angles.txt"
        if not dst.exists():
            shutil.copy(src, dst)


# ------------------------------------------------------- Sampling repo steps
def run_stats(staging: Path, sampling_repo: Path, python: str,
              workers: int) -> None:
    gen = sampling_repo / "Dream3D_generation"
    cmd = [python, str(gen / "03_copula_to_dream3d_stats.py"),
           "--in-root", str((staging / "copula").resolve()),
           "--out-root", str((staging / "stats").resolve()),
           "--workers", str(workers)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=gen)


def stats_complete(rows: list[dict], staging: Path) -> bool:
    for row in rows:
        mat, sid = row["material"], row["sample_id"]
        d = staging / "stats" / mat
        if not ((d / f"{mat}_{sid}_stats.txt").exists()
                and (d / f"{mat}_{sid}_stats_aspect_beta.txt").exists()):
            return False
    return True


def run_dream3d(staging: Path, sampling_repo: Path, python: str,
                workers: int) -> None:
    gen = sampling_repo / "Dream3D_generation"
    env = dict(os.environ,
               RVE_STATS_ROOT=str((staging / "stats").resolve()),
               RVE_ODF_ROOT=str((staging / "odf").resolve()))
    cmd = [python, str(gen / "04_generate_rves.py"),
           "--local-staging", str((staging / "rves_raw").resolve()),
           "--no-transfer",
           "--workers", str(workers)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=gen, env=env)


# --------------------------------------------------------------- collection
def collect(rows: list[dict], staging: Path, rve_root: Path,
            manifest_name: str = "manifest.csv") -> list[str]:
    """rve_root/{cid}/rve/{cid}__{um}_{scale}_{imgidx}.dream3d + manifest."""
    missing, manifest_rows = [], []
    for row in rows:
        cid, mat, sid = row["condition_id"], row["material"], row["sample_id"]
        suffix = row["image_stem"][len(cid) + 1:]  # "{um}_{scale}_{imgidx}"
        src = staging / "rves_raw" / mat / str(sid) / f"{mat}_{sid}.dream3d"
        dst = rve_root / cid / "rve" / f"{cid}__{suffix}.dream3d"
        if not src.exists():
            missing.append(row["image_stem"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest_rows.append({"condition_id": cid, "material": mat,
                              "sample_id": sid,
                              "image_stem": row["image_stem"],
                              "path": str(dst)})
    rve_root.mkdir(parents=True, exist_ok=True)
    with open(rve_root / manifest_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition_id", "material",
                                          "sample_id", "image_stem", "path"])
        w.writeheader()
        w.writerows(manifest_rows)
    return missing


def main(config_path: str = "configs/gnn.yaml") -> None:
    import pandas as pd
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    paths = yaml.safe_load(Path(cfg["paths_config"]).read_text())
    labels = pd.read_csv(Path(paths["data_root"]) / "labels" / "labels.csv")

    staging = Path(cfg.get("staging_root", "data/gnn/staging_perimage"))
    rve_root = Path(cfg["rve_root"])
    scale = int(cfg.get("stats_scale", 150))
    workers = int(cfg.get("generation_workers", 8))
    sampling_repo = Path(paths["sampling_repo"])
    python = paths.get("sampling_python", "python")
    stats_dir = Path(paths["training_data_root"]) / "mask_stats" / str(scale)

    rows = image_inventory(labels, stats_dir, scale)
    print(f"{len(rows)} experimental images at scale {scale} across "
          f"{len({r['condition_id'] for r in rows})} conditions")

    print("== per-image copula fits")
    failed = set(fit_copulas(rows, staging))

    print("== ODF angle files (one MTEX conversion per condition)")
    cache = Path(cfg.get("odf_cache_root", staging / "odf_cache"))
    seed = cfg.get("odf_cache_seed")
    if seed and Path(seed).is_dir():
        n = seed_odf_cache_from(Path(seed), labels, cache)
        if n:
            print(f"  reused {n} angle files from {seed}")
    conditions = sorted({r["condition_id"] for r in rows})
    no_odf = set(ensure_odf_cache(conditions, sampling_repo, cache, staging,
                                  paths.get("matlab_bin", "matlab"),
                                  paths["mtex_root"]))
    if no_odf:
        print(f"  WARNING: no ODF for {len(no_odf)} conditions: "
              f"{sorted(no_odf)}")
    rows = [r for r in rows if r["condition_id"] not in no_odf
            and r["image_stem"] not in failed]
    stage_angle_files(rows, cache, staging)

    if stats_complete(rows, staging):
        print("== stats tables present, skipping")
    else:
        print("== copula -> DREAM.3D stats tables")
        try:
            run_stats(staging, sampling_repo, python, workers)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  stats step failed ({e}); continuing with existing "
                  "stats tables")

    print("== DREAM.3D PipelineRunner")
    try:
        run_dream3d(staging, sampling_repo, python, workers)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  DREAM.3D step failed ({e}); collecting whatever exists")

    print("== collect")
    manifest_name = f"manifest_{scale}.csv"
    missing_out = collect(rows, staging, rve_root, manifest_name)
    print(f"manifest: {rve_root / manifest_name}")
    if missing_out:
        print(f"{len(missing_out)} RVEs missing (generation failures):")
        for m in missing_out:
            print(f"  {m}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gnn.yaml")
    args = ap.parse_args()
    main(args.config)
