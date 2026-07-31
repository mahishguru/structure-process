"""Baseline heads and classical regressors (strategy/01 section 8).

- ConditionalGaussian / ConditionalMDN: drop-in token heads with the same
  (log_prob, sample) interface as ConditionalSplineFlow, so the AR transformer
  can be ablated at fixed architecture (AR+gaussian, AR+mdn).
- IndependentMDN: same adapters, no autoregression (isolates the value of AR).
- Classical: XGBoost / GP per-label point regressors (Acta 2025 recipe anchor).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from icme_mg import ELEMENTS, LABEL_ORDER

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0


class ConditionalGaussian(nn.Module):
    """Per-token N(mu(h), sigma(h))."""

    def __init__(self, cond_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2))

    def _params(self, h):
        p = self.net(h)
        return p[..., 0], p[..., 1].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

    def log_prob(self, y, h):
        mu, log_sigma = self._params(h)
        return -0.5 * (((y - mu) / log_sigma.exp()).pow(2)
                       + math.log(2 * math.pi)) - log_sigma

    @torch.no_grad()
    def sample(self, h):
        mu, log_sigma = self._params(h)
        return mu + log_sigma.exp() * torch.randn_like(mu)


class ConditionalMDN(nn.Module):
    """Per-token Gaussian mixture (default 5 components)."""

    def __init__(self, cond_dim: int, hidden_dim: int = 128,
                 n_components: int = 5, dropout: float = 0.1):
        super().__init__()
        self.M = n_components
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * n_components))

    def _params(self, h):
        p = self.net(h)
        logits, mu, log_sigma = p.chunk(3, dim=-1)
        return (F.log_softmax(logits, dim=-1), mu,
                log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX))

    def log_prob(self, y, h):
        log_w, mu, log_sigma = self._params(h)
        comp = -0.5 * (((y[..., None] - mu) / log_sigma.exp()).pow(2)
                       + math.log(2 * math.pi)) - log_sigma
        return torch.logsumexp(log_w + comp, dim=-1)

    @torch.no_grad()
    def sample(self, h):
        log_w, mu, log_sigma = self._params(h)
        idx = torch.distributions.Categorical(logits=log_w).sample()
        mu_s = mu.gather(-1, idx[..., None])[..., 0]
        sig_s = log_sigma.gather(-1, idx[..., None])[..., 0].exp()
        return mu_s + sig_s * torch.randn_like(mu_s)


TOKEN_HEADS = {"gaussian": ConditionalGaussian, "mdn": ConditionalMDN}


class IndependentMDN(nn.Module):
    """No autoregression: adapter tokens -> pooled MLP trunk -> per-label MDN +
    element presence logits. Same losses and label space as the flow head."""

    def __init__(self, dim: int = 128, hidden_dim: int = 256,
                 n_components: int = 5, dropout: float = 0.15,
                 n_alloy_classes: int = 14):
        super().__init__()
        n_tokens = len(LABEL_ORDER)
        n_elem = len(ELEMENTS)
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.mdns = nn.ModuleList([
            ConditionalMDN(hidden_dim, 128, n_components, dropout)
            for _ in range(n_tokens)])
        self.presence_head = nn.Linear(hidden_dim, n_elem)
        self.aux_class_head = nn.Linear(hidden_dim, n_alloy_classes)

    def forward(self, z, present, log_det, cond):
        n_elem = len(ELEMENTS)
        t = self.trunk(cond.mean(dim=1))
        presence_logits = self.presence_head(t)
        nll_tokens = torch.zeros_like(z)
        for k, mdn in enumerate(self.mdns):
            lp = mdn.log_prob(z[:, k], t) + log_det[:, k]
            if k < n_elem:
                logit = presence_logits[:, k]
                nll_tokens[:, k] = torch.where(
                    present[:, k], -(F.logsigmoid(logit) + lp),
                    -F.logsigmoid(-logit))
            else:
                nll_tokens[:, k] = -lp
        return {"nll": nll_tokens.sum(dim=1), "nll_tokens": nll_tokens,
                "presence_logits": presence_logits,
                "aux_class_logits": self.aux_class_head(t)}

    @torch.no_grad()
    def sample(self, cond, n_samples: int = 256):
        n_tokens, n_elem = len(LABEL_ORDER), len(ELEMENTS)
        B, S = cond.shape[0], n_samples
        t = self.trunk(cond.mean(dim=1)).repeat_interleave(S, dim=0)
        z = torch.zeros(B * S, n_tokens, device=cond.device)
        present = torch.ones(B * S, n_elem, dtype=torch.bool, device=cond.device)
        logp = torch.zeros(B * S, device=cond.device)
        presence_logits = self.presence_head(t)
        for k, mdn in enumerate(self.mdns):
            if k < n_elem:
                logit = presence_logits[:, k]
                pres = torch.rand_like(logit) < torch.sigmoid(logit)
                present[:, k] = pres
                logp += torch.where(pres, F.logsigmoid(logit),
                                    F.logsigmoid(-logit))
                val = mdn.sample(t)
                z[:, k] = torch.where(pres, val, torch.zeros_like(val))
                lp = mdn.log_prob(z[:, k], t)
                logp += torch.where(pres, lp, torch.zeros_like(lp))
            else:
                z[:, k] = mdn.sample(t)
                logp += mdn.log_prob(z[:, k], t)
        return {"z": z.view(B, S, n_tokens),
                "present": present.view(B, S, n_elem),
                "logp": logp.view(B, S)}


# --------------------------------------------------------------------- classic
def fit_classical(X_train: np.ndarray, Y_train: np.ndarray, kind: str = "xgboost",
                  seed: int = 0) -> list:
    """Per-label point regressors on the flat descriptor vector.
    Y columns follow LABEL_ORDER. Returns one fitted model per label."""
    models = []
    for k in range(Y_train.shape[1]):
        if kind == "xgboost":
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=seed)
        elif kind == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            m = GaussianProcessRegressor(
                kernel=RBF(length_scale=10.0) + WhiteKernel(1e-2),
                normalize_y=True, random_state=seed)
        else:
            raise ValueError(kind)
        m.fit(X_train, Y_train[:, k])
        models.append(m)
    return models


def predict_classical(models: list, X: np.ndarray) -> np.ndarray:
    return np.stack([m.predict(X) for m in models], axis=1)
