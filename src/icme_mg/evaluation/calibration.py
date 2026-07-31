"""Calibration diagnostics for the probabilistic heads (strategy/02): quantile
calibration error (ECE) and PIT histograms from posterior samples."""

from __future__ import annotations

import numpy as np

from icme_mg import ELEMENTS, LABEL_ORDER


def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Probability integral transform per condition and label.
    y_true: (B, 10) physical, samples: (B, S, 10) physical (zeros applied for
    absent elements). Entries where the true element is absent are NaN.
    PIT = fraction of samples below the truth; uniform if calibrated."""
    B, S, K = samples.shape
    pit = np.full((B, K), np.nan)
    for k, name in enumerate(LABEL_ORDER):
        if name in ELEMENTS:
            valid = y_true[:, k] > 0
        else:
            valid = np.ones(B, bool)
        below = (samples[:, :, k] < y_true[:, None, k]).mean(axis=1)
        tie = (samples[:, :, k] == y_true[:, None, k]).mean(axis=1)
        pit[valid, k] = (below + 0.5 * tie)[valid]
    return pit


def quantile_ece(pit: np.ndarray, n_levels: int = 19) -> dict:
    """Expected calibration error: mean |coverage(q) - q| over quantile levels
    q in {0.05..0.95}. Returned per label plus macro average."""
    levels = np.linspace(0.05, 0.95, n_levels)
    out = {}
    for k, name in enumerate(LABEL_ORDER):
        p = pit[:, k]
        p = p[~np.isnan(p)]
        if len(p) < 2:
            out[name] = float("nan")
            continue
        cov = np.array([(p <= q).mean() for q in levels])
        out[name] = float(np.abs(cov - levels).mean())
    vals = [v for v in out.values() if np.isfinite(v)]
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def pit_histogram(pit: np.ndarray, n_bins: int = 10) -> dict:
    """Per-label PIT histogram (density heights); flat = calibrated."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = {}
    for k, name in enumerate(LABEL_ORDER):
        p = pit[:, k]
        p = p[~np.isnan(p)]
        if len(p) == 0:
            out[name] = None
            continue
        h, _ = np.histogram(p, bins=edges, density=True)
        out[name] = h.tolist()
    return out


def interval_coverage(y_true: np.ndarray, samples: np.ndarray,
                      alpha: float = 0.1) -> dict:
    """Empirical coverage of central (1-alpha) sample intervals per label."""
    lo = np.quantile(samples, alpha / 2, axis=1)
    hi = np.quantile(samples, 1 - alpha / 2, axis=1)
    out = {}
    for k, name in enumerate(LABEL_ORDER):
        valid = y_true[:, k] > 0 if name in ELEMENTS else np.ones(
            len(y_true), bool)
        if valid.sum() < 2:
            out[name] = float("nan")
            continue
        inside = ((y_true[valid, k] >= lo[valid, k])
                  & (y_true[valid, k] <= hi[valid, k]))
        out[name] = float(inside.mean())
    return out
