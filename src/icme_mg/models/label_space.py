"""Label normalization with exact change-of-variables bookkeeping.

Maps physical labels to normalized space for the flows and accumulates the
log|dz/dy| Jacobian so reported NLL is in physical units (strategy/01 section 4).

Per-label maps:
    element wt% (nonzero part): z = (log(w + 0.1) - mu_e) / sigma_e
    T_ext:                      z = (T - 350) / 150
    v_ext:                      z = (log v - mu_v) / sigma_v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from icme_mg import ELEMENTS, LABEL_ORDER

W_OFFSET = 0.1
T_CENTER, T_SCALE = 350.0, 150.0


class LabelNormalizer:
    def __init__(self, stats: dict):
        self.stats = stats

    @classmethod
    def fit(cls, labels_df) -> "LabelNormalizer":
        stats: dict = {"elements": {}}
        for e in ELEMENTS:
            w = labels_df[e].to_numpy(float)
            nz = w[w > 0]
            if len(nz) >= 2:
                lw = np.log(nz + W_OFFSET)
                mu, sigma = float(lw.mean()), float(max(lw.std(), 1e-3))
            elif len(nz) == 1:
                mu, sigma = float(np.log(nz[0] + W_OFFSET)), 0.5
            else:
                mu, sigma = float(np.log(W_OFFSET)), 1.0
            stats["elements"][e] = {"mu": mu, "sigma": sigma}
        lv = np.log(labels_df["v_ext"].to_numpy(float))
        stats["v"] = {"mu": float(lv.mean()), "sigma": float(max(lv.std(), 1e-3))}
        return cls(stats)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LabelNormalizer":
        return cls(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------- transforms
    def normalize(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """y: (B, 10) physical labels in LABEL_ORDER.
        Returns (z, log_det) where log_det[b, k] = log|dz_k/dy_k| (0 where the
        label is an absent element; hurdle handles that token's mass)."""
        z = torch.zeros_like(y)
        log_det = torch.zeros_like(y)
        for k, name in enumerate(LABEL_ORDER):
            col = y[:, k]
            if name in ELEMENTS:
                s = self.stats["elements"][name]
                pos = col > 0
                w = col.clamp(min=1e-8)
                z[:, k] = torch.where(
                    pos, (torch.log(w + W_OFFSET) - s["mu"]) / s["sigma"],
                    torch.zeros_like(col))
                log_det[:, k] = torch.where(
                    pos, -torch.log(w + W_OFFSET) - np.log(s["sigma"]),
                    torch.zeros_like(col))
            elif name == "T_ext":
                z[:, k] = (col - T_CENTER) / T_SCALE
                log_det[:, k] = -np.log(T_SCALE)
            elif name == "v_ext":
                s = self.stats["v"]
                z[:, k] = (torch.log(col) - s["mu"]) / s["sigma"]
                log_det[:, k] = -torch.log(col) - np.log(s["sigma"])
        return z, log_det

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 10) normalized -> physical (element zeros must be applied by
        the caller from the presence samples)."""
        y = torch.zeros_like(z)
        for k, name in enumerate(LABEL_ORDER):
            col = z[:, k]
            if name in ELEMENTS:
                s = self.stats["elements"][name]
                y[:, k] = (torch.exp(col * s["sigma"] + s["mu"]) - W_OFFSET
                           ).clamp(min=0.0)
            elif name == "T_ext":
                y[:, k] = col * T_SCALE + T_CENTER
            elif name == "v_ext":
                s = self.stats["v"]
                y[:, k] = torch.exp(col * s["sigma"] + s["mu"])
        return y


def dequantize(y: torch.Tensor, generator: torch.Generator | None = None,
               sigma_T: float = 5.0, rel_sigma_v: float = 0.03,
               rel_sigma_w: float = 0.02) -> torch.Tensor:
    """Tolerance-based jitter in physical space (strategy/01 section 5).
    Applied identically to flow targets and teacher-forced inputs."""
    out = y.clone()
    noise = torch.randn(y.shape, generator=generator, device=y.device)
    for k, name in enumerate(LABEL_ORDER):
        if name in ELEMENTS:
            pos = out[:, k] > 0
            out[:, k] = torch.where(
                pos, (out[:, k] * (1 + rel_sigma_w * noise[:, k])).clamp(min=1e-4),
                out[:, k])
        elif name == "T_ext":
            out[:, k] = out[:, k] + sigma_T * noise[:, k]
        elif name == "v_ext":
            out[:, k] = out[:, k] * torch.exp(rel_sigma_v * noise[:, k])
    return out
