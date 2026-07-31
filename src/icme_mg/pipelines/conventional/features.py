"""Loading and flattening of raw conventional descriptors into 1-D vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-12


def load_2point(path: str | Path) -> np.ndarray:
    """(255, 255, 3) 2-point statistics -> flat float32 vector."""
    return np.load(path).astype(np.float32).ravel()


def load_3point(path: str | Path) -> np.ndarray:
    """(4, 481, 481) mcrpy 3-point statistics -> flat float32 vector."""
    return np.load(path).astype(np.float32).ravel()


def load_gram(path: str | Path) -> np.ndarray:
    """npz of 5 Gram matrices -> per-layer Frobenius-normalized, concatenated."""
    z = np.load(path)
    parts = []
    for key in sorted(z.files):
        g = z[key].astype(np.float32)
        g = g / (np.linalg.norm(g) + EPS)
        parts.append(g.ravel())
    return np.concatenate(parts)


def load_histogram(path: str | Path) -> np.ndarray:
    """(29,) count histogram -> probability-normalized float32."""
    h = np.load(path).astype(np.float32)
    return h / (h.sum() + EPS)


def load_gsh(path: str | Path) -> np.ndarray:
    """(1102,) GSH coefficient vector -> float32."""
    return np.load(path).astype(np.float32).ravel()


LOADERS = {
    "2point": load_2point,
    "3point": load_3point,
    "gram": load_gram,
    "aspect_ratio": load_histogram,
    "eq_diameter": load_histogram,
    "gsh": load_gsh,
}
