"""Conditional 1-D rational-quadratic spline flow (Durkan et al., Neural Spline
Flows). Each continuous label token gets a flow over its normalized scalar:

    y_norm --spline_1--> --affine--> --spline_2--> u ~ N(0, 1)

All spline/affine parameters are emitted per-sample from a conditioning vector
(the transformer hidden state), so log_prob and sample are exact and cheap.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_BIN = 1e-3
MIN_DERIV = 1e-3
# softplus(x + offset) == 1 - MIN_DERIV at x = 0, so zero-init => identity slope
_DERIV_OFFSET = math.log(math.expm1(1.0 - MIN_DERIV))


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnorm_widths: torch.Tensor,
    unnorm_heights: torch.Tensor,
    unnorm_derivs: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monotone RQ spline with linear tails. inputs: (...,); params: (..., K)/(K-1).
    Returns (outputs, log_abs_det_jacobian) with identical shape to inputs."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    if not inside.any():
        return outputs, logabsdet

    x = inputs[inside]
    K = unnorm_widths.shape[-1]
    w = unnorm_widths[inside]
    h = unnorm_heights[inside]
    d = unnorm_derivs[inside]

    widths = F.softmax(w, dim=-1)
    widths = MIN_BIN + (1 - MIN_BIN * K) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0))
    cumwidths = (cumwidths * 2 - 1) * tail_bound
    cumwidths[..., 0], cumwidths[..., -1] = -tail_bound, tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(h, dim=-1)
    heights = MIN_BIN + (1 - MIN_BIN * K) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0))
    cumheights = (cumheights * 2 - 1) * tail_bound
    cumheights[..., 0], cumheights[..., -1] = -tail_bound, tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivs = MIN_DERIV + F.softplus(d + _DERIV_OFFSET)    # (.., K-1) interior
    derivs = F.pad(derivs, (1, 1), value=1.0)              # boundary derivs = 1

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, x)[..., None]
    bin_idx = bin_idx.clamp(0, K - 1)

    in_cumw = cumwidths.gather(-1, bin_idx)[..., 0]
    in_w = widths.gather(-1, bin_idx)[..., 0]
    in_cumh = cumheights.gather(-1, bin_idx)[..., 0]
    in_h = heights.gather(-1, bin_idx)[..., 0]
    delta = in_h / in_w
    d_lo = derivs.gather(-1, bin_idx)[..., 0]
    d_hi = derivs.gather(-1, (bin_idx + 1).clamp(max=K))[..., 0]

    if inverse:
        y_rel = x - in_cumh
        a = in_h * (delta - d_lo) + y_rel * (d_lo + d_hi - 2 * delta)
        b = in_h * d_lo - y_rel * (d_lo + d_hi - 2 * delta)
        c = -delta * y_rel
        disc = b.pow(2) - 4 * a * c
        disc = disc.clamp(min=0)
        theta = (2 * c) / (-b - torch.sqrt(disc))
        theta = theta.clamp(0, 1)
        out = theta * in_w + in_cumw
        denom = delta + (d_lo + d_hi - 2 * delta) * theta * (1 - theta)
        deriv_num = delta.pow(2) * (
            d_hi * theta.pow(2) + 2 * delta * theta * (1 - theta)
            + d_lo * (1 - theta).pow(2))
        lad = -(torch.log(deriv_num) - 2 * torch.log(denom))
    else:
        theta = ((x - in_cumw) / in_w).clamp(0, 1)
        num = in_h * (delta * theta.pow(2) + d_lo * theta * (1 - theta))
        denom = delta + (d_lo + d_hi - 2 * delta) * theta * (1 - theta)
        out = in_cumh + num / denom
        deriv_num = delta.pow(2) * (
            d_hi * theta.pow(2) + 2 * delta * theta * (1 - theta)
            + d_lo * (1 - theta).pow(2))
        lad = torch.log(deriv_num) - 2 * torch.log(denom)

    outputs[inside] = out
    logabsdet[inside] = lad
    return outputs, logabsdet


class ConditionalSplineFlow(nn.Module):
    """Two RQ spline layers with an affine layer in between, all conditioned on h.

    log_prob(y_norm | h): exact. sample(h): exact inverse.
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 128, n_bins: int = 8,
                 tail_bound: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.K = n_bins
        self.tail_bound = tail_bound
        n_spline = 3 * n_bins - 1
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, 2 * n_spline + 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # start at identity transform

    def _params(self, h: torch.Tensor):
        p = self.head(self.trunk(h))
        K = self.K
        n = 3 * K - 1
        s1, s2 = p[..., :n], p[..., n:2 * n]
        log_scale = torch.tanh(p[..., 2 * n]) * 3.0     # bounded affine scale
        shift = p[..., 2 * n + 1]

        def split(s):
            return s[..., :K], s[..., K:2 * K], s[..., 2 * K:]

        return split(s1), split(s2), log_scale, shift

    def log_prob(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """y: (B,) normalized scalar; h: (B, cond_dim) -> (B,) log density."""
        (w1, h1, d1), (w2, h2, d2), log_scale, shift = self._params(h)
        z, lad1 = rational_quadratic_spline(y, w1, h1, d1, False, self.tail_bound)
        z = z * torch.exp(log_scale) + shift
        lad_aff = log_scale
        z, lad2 = rational_quadratic_spline(z, w2, h2, d2, False, self.tail_bound)
        log_base = -0.5 * (z.pow(2) + math.log(2 * math.pi))
        return log_base + lad1 + lad_aff + lad2

    @torch.no_grad()
    def sample(self, h: torch.Tensor) -> torch.Tensor:
        (w1, h1, d1), (w2, h2, d2), log_scale, shift = self._params(h)
        u = torch.randn(h.shape[:-1], device=h.device)
        z, _ = rational_quadratic_spline(u, w2, h2, d2, True, self.tail_bound)
        z = (z - shift) * torch.exp(-log_scale)
        y, _ = rational_quadratic_spline(z, w1, h1, d1, True, self.tail_bound)
        return y
