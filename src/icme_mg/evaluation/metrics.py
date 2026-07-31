"""Evaluation metrics (strategy/02): per-label point accuracy, presence F1,
joint NLL, alloy identification top-k, paired significance tests, and point
estimation from posterior samples (joint-MAP + k-medoids shortlist)."""

from __future__ import annotations

import numpy as np
import torch

from icme_mg import ELEMENTS, LABEL_ORDER

N_ELEM = len(ELEMENTS)


# ------------------------------------------------------------- point estimates
def joint_map_estimate(samples: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the highest joint log-prob sample per condition.
    samples: output of head.sample() with z (B,S,10), present (B,S,8),
    logp (B,S). Returns (z_map (B,10), present_map (B,8))."""
    idx = samples["logp"].argmax(dim=1)
    B = idx.shape[0]
    ar = torch.arange(B)
    return samples["z"][ar, idx], samples["present"][ar, idx]


def kmedoids_shortlist(y_phys: torch.Tensor, k: int = 5, iters: int = 20,
                       seed: int = 0) -> torch.Tensor:
    """Diverse candidate shortlist from posterior samples of ONE condition.
    y_phys: (S, 10) physical-unit samples. Returns (k, 10) medoids."""
    S = y_phys.shape[0]
    x = (y_phys - y_phys.mean(0)) / (y_phys.std(0) + 1e-8)
    d = torch.cdist(x, x)
    g = torch.Generator().manual_seed(seed)
    med = torch.randperm(S, generator=g)[:k]
    for _ in range(iters):
        assign = d[:, med].argmin(dim=1)
        new = med.clone()
        for j in range(k):
            members = (assign == j).nonzero(as_tuple=True)[0]
            if len(members):
                sub = d[members][:, members]
                new[j] = members[sub.sum(1).argmin()]
        if torch.equal(new, med):
            break
        med = new
    return y_phys[med]


# --------------------------------------------------------------- point metrics
def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/MAPE/R2 per label. For elements, MAE over all conditions but MAPE/R2
    only where the element is truly present (zeros make MAPE undefined)."""
    out = {}
    for k, name in enumerate(LABEL_ORDER):
        t, p = y_true[:, k], y_pred[:, k]
        mask = t > 0 if name in ELEMENTS else np.ones_like(t, bool)
        m = {"mae": float(np.abs(t - p).mean())}
        if mask.sum() >= 2:
            tm, pm = t[mask], p[mask]
            m["mape"] = float((np.abs(tm - pm) / np.abs(tm)).mean() * 100)
            ss_res = float(((tm - pm) ** 2).sum())
            ss_tot = float(((tm - tm.mean()) ** 2).sum())
            m["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[name] = m
    return out


def presence_f1(true_present: np.ndarray, pred_present: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(ELEMENTS):
        t, p = true_present[:, k], pred_present[:, k]
        tp = float((t & p).sum())
        prec = tp / max(p.sum(), 1)
        rec = tp / max(t.sum(), 1)
        out[name] = 2 * prec * rec / max(prec + rec, 1e-8)
    out["macro"] = float(np.mean(list(out.values())))
    return out


# --------------------------------------------------------- alloy identification
def alloy_topk(y_pred: np.ndarray, nominal: np.ndarray, alloy_names: list[str],
               true_alloys: list[str], ks: tuple[int, ...] = (1, 3)) -> dict:
    """Nearest nominal composition (z-scored element-space distance).
    nominal: (A, 8) per-alloy compositions in ELEMENTS order."""
    mu, sd = nominal.mean(0), nominal.std(0) + 1e-8
    d = np.stack(
        [np.linalg.norm((y_pred[:, :N_ELEM] - mu) / sd - (a - mu) / sd, axis=1)
         for a in nominal], axis=1)
    order = np.argsort(d, axis=1)
    out = {}
    for k in ks:
        hits = [true_alloys[i] in [alloy_names[j] for j in order[i, :k]]
                for i in range(len(true_alloys))]
        out[f"top{k}"] = float(np.mean(hits))
    return out


# ------------------------------------------------------------------ statistics
def paired_wilcoxon(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """p-value of the paired Wilcoxon signed-rank test on per-condition errors
    (a vs b). Small p with median(a-b)<0 favors model a."""
    from scipy.stats import wilcoxon
    diff = errors_a - errors_b
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(errors_a, errors_b).pvalue)


def aggregate_per_condition(condition_ids: list[str], values: np.ndarray
                            ) -> tuple[list[str], np.ndarray]:
    """Average instance-level values to condition level (evaluation unit)."""
    order: dict[str, list[int]] = {}
    for i, c in enumerate(condition_ids):
        order.setdefault(c, []).append(i)
    cids = list(order)
    agg = np.stack([values[order[c]].mean(axis=0) for c in cids])
    return cids, agg
