"""Lattice-aware prediction heads for the discrete composition label space.

The corpus contains only 14 distinct alloy compositions; element amounts live
on a small discrete lattice (e.g. Gd in {0, 2, 5, 10} wt.%). Continuous flow
heads spread probability mass between lattice points, inflating element MAE.
These heads exploit the lattice structure while keeping exact likelihoods and
the same (forward / sample) interface as FlowTransformerHead:

    PrototypeMixtureHead ("proto_mix"): composition is a mixture over the
        training alloys' nominal compositions (per-prototype hurdle model with
        epsilon presence smoothing so LOAO stays finite). Process labels keep
        conditional spline flows.

    LevelMixtureHead ("level_mix"): per-element hurdle + categorical mixture
        of Gaussians with means fixed at the element's observed training
        levels (in normalized space). Independent across elements given the
        pooled conditioning. Process labels keep conditional spline flows.

Both operate on NORMALIZED labels and add the normalizer's log-Jacobian so the
reported NLL is in physical units, exactly like the flow head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from icme_mg import ELEMENTS, LABEL_ORDER

N_TOKENS = len(LABEL_ORDER)     # 10
N_ELEM = len(ELEMENTS)          # 7
LOG_2PI = float(np.log(2 * np.pi))


def _gauss_logpdf(x: torch.Tensor, mu: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((x - mu) / sigma) ** 2 + LOG_2PI) - torch.log(sigma)


class _Trunk(nn.Module):
    """Conditioning tokens -> feature vector for logits and flows.
    pool: "mean" or "attn" (learned-query attention pooling)."""

    def __init__(self, dim: int, dropout: float = 0.1, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        if pool == "attn":
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, 4, dropout=dropout,
                                              batch_first=True)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.GELU())

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if self.pool == "attn":
            q = self.query.expand(cond.shape[0], -1, -1)
            pooled, _ = self.attn(q, cond, cond, need_weights=False)
            pooled = pooled[:, 0] + cond.mean(dim=1)
        else:
            pooled = cond.mean(dim=1)
        return self.net(pooled)


class PrototypeMixtureHead(nn.Module):
    """p(elements | cond) = sum_k w_k(cond) * hurdle-Gaussian(prototype_k).

    prototypes_z: (K, 8) normalized nominal element values of the TRAINING
    alloys (0 where absent); presence_pat: (K, 8) bool; proto_class_idx: (K,)
    global alloy-class indices for the aux logits.
    """

    def __init__(self, dim: int, prototypes_z: torch.Tensor,
                 presence_pat: torch.Tensor, proto_class_idx: torch.Tensor,
                 n_alloy_classes: int = 14, eps: float = 1e-2,
                 sigma_init: float = 0.1, bg_sigma: float = 3.0,
                 flow_bins: int = 8, flow_hidden: int = 128,
                 dropout: float = 0.1, **_):
        super().__init__()
        from .spline_flow import ConditionalSplineFlow
        self.register_buffer("proto_z", prototypes_z.float())        # (K, 8)
        self.register_buffer("proto_pres", presence_pat.bool())       # (K, 8)
        self.register_buffer("proto_cls", proto_class_idx.long())     # (K,)
        self.n_alloy_classes = n_alloy_classes
        self.eps, self.bg_sigma = eps, bg_sigma
        K = prototypes_z.shape[0]

        self.trunk = _Trunk(dim, dropout)
        self.proto_logits = nn.Linear(dim, K)
        # per-element lattice noise, softplus-parameterized
        self.raw_sigma = nn.Parameter(
            torch.full((N_ELEM,), float(np.log(np.expm1(sigma_init)))))
        self.flow_T = ConditionalSplineFlow(dim, flow_hidden, flow_bins)
        self.flow_v = ConditionalSplineFlow(dim, flow_hidden, flow_bins)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 0.02                      # (8,)

    def _element_logp(self, z: torch.Tensor, present: torch.Tensor,
                      log_det: torch.Tensor, h: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (comp_logp (B,), proto_log_post (B, K))."""
        logw = F.log_softmax(self.proto_logits(h), dim=-1)             # (B, K)
        z_e = z[:, :N_ELEM].unsqueeze(1)                               # (B,1,8)
        ld_e = log_det[:, :N_ELEM].unsqueeze(1)                        # (B,1,8)
        obs_pres = present.unsqueeze(1)                                # (B,1,8)
        pp = self.proto_pres.unsqueeze(0)                              # (1,K,8)
        sig = self.sigma.view(1, 1, -1)
        mu = self.proto_z.unsqueeze(0)                                 # (1,K,8)

        log_pi = torch.where(pp, torch.tensor(np.log1p(-self.eps),
                                              device=z.device),
                             torch.tensor(np.log(self.eps), device=z.device))
        log_1mpi = torch.where(pp, torch.tensor(np.log(self.eps),
                                                device=z.device),
                               torch.tensor(np.log1p(-self.eps),
                                            device=z.device))
        dens_proto = _gauss_logpdf(z_e, mu, sig)                       # match
        dens_bg = _gauss_logpdf(z_e, torch.zeros_like(z_e),
                                torch.full_like(z_e, self.bg_sigma))
        dens = torch.where(pp, dens_proto, dens_bg)
        per_elem = torch.where(obs_pres, log_pi + dens + ld_e, log_1mpi)
        log_lik = per_elem.sum(dim=-1)                                 # (B, K)
        joint = logw + log_lik
        comp_logp = torch.logsumexp(joint, dim=-1)                     # (B,)
        return comp_logp, joint - comp_logp.unsqueeze(-1)

    def forward(self, z, present, log_det, cond) -> dict:
        h = self.trunk(cond)
        comp_logp, _ = self._element_logp(z, present, log_det, h)
        lp_T = self.flow_T.log_prob(z[:, N_ELEM], h) + log_det[:, N_ELEM]
        lp_v = self.flow_v.log_prob(z[:, N_ELEM + 1], h) + log_det[:, N_ELEM + 1]
        nll = -(comp_logp + lp_T + lp_v)

        logw = F.log_softmax(self.proto_logits(h), dim=-1)             # (B, K)
        # presence marginal: p(e) = sum_k w_k * pi_ke
        w = logw.exp()
        pi = torch.where(self.proto_pres.unsqueeze(0),
                         torch.tensor(1 - self.eps, device=z.device),
                         torch.tensor(self.eps, device=z.device))
        p_e = (w.unsqueeze(-1) * pi).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        presence_logits = torch.log(p_e) - torch.log1p(-p_e)

        aux = torch.full((z.shape[0], self.n_alloy_classes), -1e4,
                         device=z.device)
        aux = aux.scatter(1, self.proto_cls.unsqueeze(0).expand(
            z.shape[0], -1), logw)
        return {"nll": nll, "presence_logits": presence_logits,
                "aux_class_logits": aux}

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 256) -> dict:
        B, S = cond.shape[0], n_samples
        h = self.trunk(cond)                                           # (B, dim)
        logw = F.log_softmax(self.proto_logits(h), dim=-1)             # (B, K)
        k = torch.multinomial(logw.exp(), S, replacement=True)         # (B, S)
        mu = self.proto_z[k]                                           # (B,S,8)
        pres = self.proto_pres[k]                                      # (B,S,8)
        sig = self.sigma.view(1, 1, -1)
        z_e = mu + sig * torch.randn_like(mu)
        z_e = torch.where(pres, z_e, torch.zeros_like(z_e))

        h_rep = h.repeat_interleave(S, dim=0)
        z_T = self.flow_T.sample(h_rep)
        z_v = self.flow_v.sample(h_rep)
        lp_T = self.flow_T.log_prob(z_T, h_rep).view(B, S)
        lp_v = self.flow_v.log_prob(z_v, h_rep).view(B, S)

        logp = (logw.gather(1, k)
                + torch.where(pres, _gauss_logpdf(z_e, mu, sig),
                              torch.zeros_like(z_e)).sum(-1)
                + lp_T + lp_v)
        z = torch.zeros(B, S, N_TOKENS, device=cond.device)
        z[..., :N_ELEM] = z_e
        z[..., N_ELEM] = z_T.view(B, S)
        z[..., N_ELEM + 1] = z_v.view(B, S)
        return {"z": z, "present": pres, "logp": logp}


class LevelMixtureHead(nn.Module):
    """Per-element hurdle + mixture of Gaussians at the observed training
    levels (normalized space). levels_z: list of 8 float tensors, the distinct
    nonzero normalized values seen in training for each element.
    """

    def __init__(self, dim: int, levels_z: list[torch.Tensor],
                 n_alloy_classes: int = 14, sigma_init: float = 0.1,
                 flow_bins: int = 8, flow_hidden: int = 128,
                 dropout: float = 0.1, **_):
        super().__init__()
        from .spline_flow import ConditionalSplineFlow
        self.trunk = _Trunk(dim, dropout)
        for e, lv in enumerate(levels_z):
            self.register_buffer(f"levels_{e}", lv.float())
        self.n_levels = [len(lv) for lv in levels_z]
        self.presence_head = nn.Linear(dim, N_ELEM)
        self.level_logits = nn.ModuleList(
            [nn.Linear(dim, max(n, 1)) for n in self.n_levels])
        self.raw_sigma = nn.Parameter(
            torch.full((N_ELEM,), float(np.log(np.expm1(sigma_init)))))
        self.flow_T = ConditionalSplineFlow(dim, flow_hidden, flow_bins)
        self.flow_v = ConditionalSplineFlow(dim, flow_hidden, flow_bins)
        self.aux_class_head = nn.Linear(dim, n_alloy_classes)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 0.02

    def _levels(self, e: int) -> torch.Tensor:
        return getattr(self, f"levels_{e}")

    def forward(self, z, present, log_det, cond) -> dict:
        h = self.trunk(cond)
        presence_logits = self.presence_head(h)                        # (B, 8)
        nll = torch.zeros(z.shape[0], device=z.device)
        sig = self.sigma
        for e in range(N_ELEM):
            logit = presence_logits[:, e]
            log_pi, log_1mpi = F.logsigmoid(logit), F.logsigmoid(-logit)
            lv = self._levels(e)                                       # (L,)
            if len(lv) == 0:
                nll = nll - torch.where(present[:, e], log_pi, log_1mpi)
                continue
            lw = F.log_softmax(self.level_logits[e](h), dim=-1)        # (B, L)
            dens = _gauss_logpdf(z[:, e:e + 1], lv.unsqueeze(0),
                                 sig[e])                               # (B, L)
            lp_val = torch.logsumexp(lw + dens, dim=-1) + log_det[:, e]
            nll = nll - torch.where(present[:, e], log_pi + lp_val, log_1mpi)
        lp_T = self.flow_T.log_prob(z[:, N_ELEM], h) + log_det[:, N_ELEM]
        lp_v = self.flow_v.log_prob(z[:, N_ELEM + 1], h) + log_det[:, N_ELEM + 1]
        nll = nll - lp_T - lp_v
        return {"nll": nll, "presence_logits": presence_logits,
                "aux_class_logits": self.aux_class_head(h)}

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 256) -> dict:
        B, S = cond.shape[0], n_samples
        h = self.trunk(cond)
        sig = self.sigma
        z = torch.zeros(B, S, N_TOKENS, device=cond.device)
        present = torch.zeros(B, S, N_ELEM, dtype=torch.bool,
                              device=cond.device)
        logp = torch.zeros(B, S, device=cond.device)
        presence_logits = self.presence_head(h)
        for e in range(N_ELEM):
            logit = presence_logits[:, e:e + 1].expand(B, S)
            pi = torch.sigmoid(logit)
            pres = torch.rand_like(pi) < pi
            present[..., e] = pres
            logp += torch.where(pres, F.logsigmoid(logit),
                                F.logsigmoid(-logit))
            lv = self._levels(e)
            if len(lv) == 0:
                continue
            lw = F.log_softmax(self.level_logits[e](h), dim=-1)        # (B, L)
            idx = torch.multinomial(lw.exp(), S, replacement=True)     # (B, S)
            val = lv[idx] + sig[e] * torch.randn(B, S, device=cond.device)
            z[..., e] = torch.where(pres, val, torch.zeros_like(val))
            dens = _gauss_logpdf(z[..., e].unsqueeze(-1), lv.view(1, 1, -1),
                                 sig[e])
            lp_val = torch.logsumexp(lw.unsqueeze(1) + dens, dim=-1)
            logp += torch.where(pres, lp_val, torch.zeros_like(lp_val))
        h_rep = h.repeat_interleave(S, dim=0)
        z_T = self.flow_T.sample(h_rep)
        z_v = self.flow_v.sample(h_rep)
        z[..., N_ELEM] = z_T.view(B, S)
        z[..., N_ELEM + 1] = z_v.view(B, S)
        logp += self.flow_T.log_prob(z_T, h_rep).view(B, S)
        logp += self.flow_v.log_prob(z_v, h_rep).view(B, S)
        return {"z": z, "present": present, "logp": logp}


# --------------------------------------------------------------- construction
def lattice_data_from_labels(labels_df, normalizer) -> dict:
    """Extract prototypes and per-element levels (normalized space) from the
    TRAINING labels. labels_df: rows = training conditions, columns include
    'alloy' and ELEMENTS. Returns tensors for both head constructors."""
    full_alloys = None  # global class order must match datasets.labels_tensor
    protos, pres, cls = [], [], []
    alloys_train = sorted(labels_df["alloy"].unique())
    for a in alloys_train:
        w = labels_df[labels_df["alloy"] == a].iloc[0][
            list(ELEMENTS)].to_numpy(float)
        protos.append(w)
        pres.append(w > 0)
    protos = np.stack(protos)                                          # (K, 8)
    pres = np.stack(pres)

    def z_of(w_col: np.ndarray, e: str) -> np.ndarray:
        s = normalizer.stats["elements"][e]
        return (np.log(w_col + 0.1) - s["mu"]) / s["sigma"]

    proto_z = np.zeros_like(protos)
    for j, e in enumerate(ELEMENTS):
        nz = protos[:, j] > 0
        proto_z[nz, j] = z_of(protos[nz, j], e)

    levels_z = []
    for j, e in enumerate(ELEMENTS):
        vals = np.unique(np.round(
            labels_df[e].to_numpy(float)[labels_df[e].to_numpy(float) > 0], 4))
        levels_z.append(torch.tensor(z_of(vals, e) if len(vals) else
                                     np.zeros(0), dtype=torch.float32))

    return {"prototypes_z": torch.tensor(proto_z, dtype=torch.float32),
            "presence_pat": torch.tensor(pres),
            "alloys_train": alloys_train,
            "levels_z": levels_z}
