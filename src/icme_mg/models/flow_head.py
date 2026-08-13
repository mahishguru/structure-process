"""Autoregressive flow-transformer head (strategy/01 sections 2-4, 6).

Sequence: [BOS] + 10 label tokens (8 elements, T_ext, v_ext), strict causal
self-attention, cross-attention to descriptor adapter tokens. Per element token:
hurdle presence logit + conditional RQ spline flow over the positive part.
T and v tokens: spline flows. Exact joint NLL; ancestral sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from icme_mg import ELEMENTS, LABEL_ORDER
from .spline_flow import ConditionalSplineFlow

N_TOKENS = len(LABEL_ORDER)          # 10
N_ELEM = len(ELEMENTS)               # 7


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + 1e-6)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(a) * b))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                               batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                batch_first=True)
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond, causal_mask):
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        a, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class FlowTransformerHead(nn.Module):
    """p(y | cond_tokens) = prod_k p(y_k | y_<k, cond).

    Inputs are NORMALIZED labels (see label_space.LabelNormalizer); element
    presence is inferred from the physical labels' zero pattern, passed as a
    boolean mask.
    """

    def __init__(self, dim: int = 128, layers: int = 4, heads: int = 4,
                 ffn_hidden: int = 256, dropout: float = 0.15,
                 flow_bins: int = 8, flow_hidden: int = 128,
                 n_alloy_classes: int = 14, cond_dropout: float = 0.1,
                 token_head: str = "spline"):
        super().__init__()
        self.dim = dim
        self.cond_dropout = cond_dropout

        self.identity_emb = nn.Parameter(torch.randn(N_TOKENS + 1, dim) * 0.02)
        self.zero_emb = nn.Parameter(torch.randn(dim) * 0.02)  # absent element
        self.value_lift = nn.ModuleList(
            [nn.Linear(1, dim) for _ in range(N_TOKENS)])
        self.bos_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList([
            DecoderBlock(dim, heads, ffn_hidden, dropout)
            for _ in range(layers)])
        self.final_norm = RMSNorm(dim)

        if token_head == "spline":
            self.flows = nn.ModuleList([
                ConditionalSplineFlow(dim, flow_hidden, flow_bins)
                for _ in range(N_TOKENS)])
        else:  # AR ablations: "gaussian" | "mdn" (baselines.TOKEN_HEADS)
            from .baselines import TOKEN_HEADS
            self.flows = nn.ModuleList([
                TOKEN_HEADS[token_head](dim, flow_hidden)
                for _ in range(N_TOKENS)])
        self.presence_head = nn.Linear(dim, N_ELEM)
        self.aux_class_head = nn.Linear(dim, n_alloy_classes)

        # Input sequence: [BOS, in_1..in_9] (label 10 is never an input) -> len 10.
        mask = torch.full((N_TOKENS, N_TOKENS), float("-inf"))
        self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

    # ------------------------------------------------------------------ core
    def _embed_inputs(self, z: torch.Tensor, present: torch.Tensor,
                      cond: torch.Tensor) -> torch.Tensor:
        """Teacher-forced input sequence [BOS, in_1..in_9]; position k's hidden
        state predicts label k (0-indexed). z: (B, 10) normalized values;
        present: (B, 8) bool."""
        B = z.shape[0]
        toks = [self.bos_proj(cond.mean(dim=1)) + self.identity_emb[0]]
        for k in range(N_TOKENS - 1):          # inputs of labels 1..9
            v = self.value_lift[k](z[:, k:k + 1])
            if k < N_ELEM:
                v = torch.where(present[:, k:k + 1], v,
                                self.zero_emb.expand(B, -1))
            toks.append(v + self.identity_emb[k + 1])
        return torch.stack(toks, dim=1)        # (B, 10, dim)

    def _hidden(self, z, present, cond):
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(cond.shape[0], 1, 1, device=cond.device)
                    > self.cond_dropout).float()
            cond = cond * keep
        x = self._embed_inputs(z, present, cond)
        for blk in self.blocks:
            x = blk(x, cond, self.causal_mask)
        return self.final_norm(x)              # (B, 10, dim); h[:, k] predicts label k

    def forward(self, z: torch.Tensor, present: torch.Tensor,
                log_det: torch.Tensor, cond: torch.Tensor) -> dict:
        """Exact NLL (physical units via log_det from the normalizer).

        z: (B, 10) normalized labels (dequantized), present: (B, 8) bool,
        log_det: (B, 10) normalization Jacobians, cond: (B, Nc, dim).
        """
        h = self._hidden(z, present, cond)      # (B, 10, dim)
        h_pred = h                              # h[:, k] predicts label k

        nll_tokens = torch.zeros_like(z)
        presence_logits = torch.zeros(z.shape[0], N_ELEM, device=z.device)
        for k in range(N_TOKENS):
            hk = h_pred[:, k]
            lp_flow = self.flows[k].log_prob(z[:, k], hk) + log_det[:, k]
            if k < N_ELEM:
                logit = self.presence_head(hk)[:, k]
                presence_logits[:, k] = logit
                log_pi = F.logsigmoid(logit)
                log_1mpi = F.logsigmoid(-logit)
                pres = present[:, k]
                nll_tokens[:, k] = torch.where(
                    pres, -(log_pi + lp_flow), -log_1mpi)
            else:
                nll_tokens[:, k] = -lp_flow

        return {
            "nll": nll_tokens.sum(dim=1),          # (B,) joint NLL, physical units
            "nll_tokens": nll_tokens,               # (B, 10)
            "presence_logits": presence_logits,     # (B, 8)
            "aux_class_logits": self.aux_class_head(cond.mean(dim=1)),
        }

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 256) -> dict:
        """Ancestral sampling. cond: (B, Nc, dim).
        Returns normalized samples z (B, S, 10), presence (B, S, 8) bool, and
        per-sample joint log-prob in normalized space (B, S)."""
        B, S = cond.shape[0], n_samples
        cond_rep = cond.repeat_interleave(S, dim=0)     # (B*S, Nc, dim)
        z = torch.zeros(B * S, N_TOKENS, device=cond.device)
        present = torch.ones(B * S, N_ELEM, dtype=torch.bool, device=cond.device)
        logp = torch.zeros(B * S, device=cond.device)

        for k in range(N_TOKENS):
            h = self._hidden(z, present, cond_rep)[:, k]
            if k < N_ELEM:
                logit = self.presence_head(h)[:, k]
                pi = torch.sigmoid(logit)
                pres_k = torch.rand_like(pi) < pi
                present[:, k] = pres_k
                logp += torch.where(pres_k, F.logsigmoid(logit),
                                    F.logsigmoid(-logit))
                val = self.flows[k].sample(h)
                z[:, k] = torch.where(pres_k, val, torch.zeros_like(val))
                lp = self.flows[k].log_prob(z[:, k], h)
                logp += torch.where(pres_k, lp, torch.zeros_like(lp))
            else:
                z[:, k] = self.flows[k].sample(h)
                logp += self.flows[k].log_prob(z[:, k], h)

        return {"z": z.view(B, S, N_TOKENS),
                "present": present.view(B, S, N_ELEM),
                "logp": logp.view(B, S)}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
