"""Grain-adjacency graph construction.

Two sources, one graph schema:

- Synthetic: DREAM.3D RVEs (.dream3d HDF5) with per-voxel FeatureIds and
  per-grain Euler angles (from the Sampling/Dream3D_generation pipeline).
- Experimental: binarized OM masks (grain boundaries = 0, interior = 1) from the
  Acta characterization pipeline; grains have no individual orientation, so
  orientation node features are zero-filled and a graph-level GSH texture vector
  can be attached instead.

Graph schema (torch_geometric.data.Data):
    x          (N, 9)  node features: [log_area, aspect_ratio, ellipse_orientation
                        (sin, cos), centroid_x, centroid_y (normalized),
                        quat_w, |quat_xyz| or zeros, boundary_fraction]
    edge_index (2, E)  undirected grain adjacency
    edge_attr  (E, 2)  [shared_boundary_length (normalized), misorientation_angle
                        (rad, 0 when orientations unavailable)]
    y_condition        condition_id string (label lookup happens in the dataset)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- helpers
def _region_props(label_img: np.ndarray) -> dict[int, dict]:
    from skimage.measure import regionprops
    props = {}
    for r in regionprops(label_img):
        a, b = r.axis_major_length, r.axis_minor_length
        props[r.label] = {
            "area": r.area,
            "aspect_ratio": (b / a) if a > 0 else 1.0,
            "orientation": r.orientation,
            "centroid": r.centroid,
            "perimeter": max(r.perimeter, 1.0),
        }
    return props


def _adjacency(label_img: np.ndarray) -> dict[tuple[int, int], int]:
    """Shared-boundary pixel counts between 4-neighbor grain pairs."""
    pairs: dict[tuple[int, int], int] = {}
    for shift_ax in (0, 1):
        a = label_img.take(range(label_img.shape[shift_ax] - 1), axis=shift_ax)
        b = label_img.take(range(1, label_img.shape[shift_ax]), axis=shift_ax)
        mask = (a != b) & (a > 0) & (b > 0)
        for u, v in zip(a[mask].ravel(), b[mask].ravel()):
            key = (min(u, v), max(u, v))
            pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Bunge ZXZ Euler angles (rad) -> quaternion (w, x, y, z)."""
    phi1, Phi, phi2 = euler[..., 0], euler[..., 1], euler[..., 2]
    c1, s1 = np.cos(phi1 / 2), np.sin(phi1 / 2)
    c, s = np.cos(Phi / 2), np.sin(Phi / 2)
    c2, s2 = np.cos(phi2 / 2), np.sin(phi2 / 2)
    return np.stack([
        c1 * c * c2 - s1 * c * s2,
        c1 * s * c2 + s1 * s * s2,
        s1 * s * c2 - c1 * s * s2,
        c1 * c * s2 + s1 * c * c2,
    ], axis=-1)


def _misorientation(q1: np.ndarray, q2: np.ndarray) -> float:
    """Misorientation angle in radians (no crystal symmetry reduction; the GNN
    consumes it as a relative feature, symmetry reduction is a refinement)."""
    dot = np.abs(np.clip(np.sum(q1 * q2), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def build_graph(label_img: np.ndarray, orientations: np.ndarray | None = None,
                min_grain_px: int = 5):
    """label_img: (H, W) int grain labels (0 = boundary/unassigned).
    orientations: (max_label + 1, 3) Bunge Euler angles per grain id, or None.
    """
    import torch
    from torch_geometric.data import Data

    h, w = label_img.shape
    props = _region_props(label_img)
    keep = sorted(l for l, p in props.items() if p["area"] >= min_grain_px)
    if len(keep) < 2:
        raise ValueError("Fewer than 2 grains after filtering")
    idx = {l: i for i, l in enumerate(keep)}

    quats = None
    if orientations is not None:
        quats = _euler_to_quat(orientations)

    feats = []
    for l in keep:
        p = props[l]
        qw, qxyz = 0.0, 0.0
        if quats is not None and l < len(quats):
            q = quats[l]
            qw, qxyz = float(abs(q[0])), float(np.linalg.norm(q[1:]))
        feats.append([
            np.log(p["area"]),
            p["aspect_ratio"],
            np.sin(2 * p["orientation"]),
            np.cos(2 * p["orientation"]),
            p["centroid"][0] / h,
            p["centroid"][1] / w,
            qw, qxyz,
            min(p["perimeter"] / p["area"], 2.0),
        ])
    x = torch.tensor(np.asarray(feats, np.float32))

    edges, eattr = [], []
    for (u, v), n_px in _adjacency(label_img).items():
        if u not in idx or v not in idx:
            continue
        mis = 0.0
        if quats is not None and u < len(quats) and v < len(quats):
            mis = _misorientation(quats[u], quats[v])
        for a, b in ((idx[u], idx[v]), (idx[v], idx[u])):
            edges.append([a, b])
            eattr.append([n_px / (h + w), mis])
    edge_index = torch.tensor(np.asarray(edges, np.int64).T)
    edge_attr = torch.tensor(np.asarray(eattr, np.float32))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------- entry points
def graph_from_dream3d(path: str | Path, min_grain_px: int = 5):
    """Read a DREAM.3D 6.5 file (300x300x1 synthetic RVE) into a grain graph.

    Layout (verified on Sampling-pipeline outputs): the volume lives in the
    data container that has CellData/FeatureIds (SyntheticVolumeDataContainer;
    a StatsGeneratorDataContainer sibling has no cell data), and per-grain
    Euler angles live in its "Grain Data" feature matrix.
    """
    import h5py

    with h5py.File(path, "r") as f:
        dc = f["DataContainers"]
        container = next((c for c in dc.values()
                          if "CellData" in c and "FeatureIds" in c["CellData"]),
                         None)
        if container is None:
            raise KeyError("no data container with CellData/FeatureIds")
        feature_ids = np.asarray(
            container["CellData"]["FeatureIds"]).squeeze().astype(np.int64)
        feat_grp = next((container[k] for k in ("CellFeatureData", "Grain Data")
                         if k in container and "EulerAngles" in container[k]),
                        None)
        eulers = np.asarray(feat_grp["EulerAngles"]) if feat_grp is not None \
            else None
    if feature_ids.ndim != 2:
        feature_ids = feature_ids.reshape(feature_ids.shape[-2], feature_ids.shape[-1])
    return build_graph(feature_ids, orientations=eulers, min_grain_px=min_grain_px)


def graph_from_mask(path: str | Path, min_grain_px: int = 5):
    """Binarized OM mask (.npy or image; boundaries 0 / interior 1) -> grain graph."""
    from skimage.measure import label as cc_label

    p = Path(path)
    if p.suffix == ".npy":
        binary = np.load(p)
    else:
        from PIL import Image
        binary = np.asarray(Image.open(p).convert("L"))
    binary = (binary > binary.max() / 2).astype(np.int32)
    labels = cc_label(binary, connectivity=1)
    return build_graph(labels, orientations=None, min_grain_px=min_grain_px)
