"""Descriptor-pipeline adapters: map each representation to conditioning tokens
(B, Nc, dim) consumed by the flow head's cross-attention (strategy/01, 2.1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VectorAdapter(nn.Module):
    """Flat descriptor vector -> n_tokens x dim (conventional pipeline)."""

    def __init__(self, in_dim: int, dim: int = 128, n_tokens: int = 8):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_dim, n_tokens * dim)
        self.pos = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.proj(x).view(B, self.n_tokens, -1)
        return self.norm(t + self.pos)


class TokenAdapter(nn.Module):
    """Pre-tokenized latent (B, n_tokens, token_dim) -> (B, n_tokens, dim).
    GenAI pipeline: 16 spatial tokens x 80-D from ViT-FMDiT-1280."""

    def __init__(self, token_dim: int = 80, dim: int = 128, n_tokens: int = 16):
        super().__init__()
        self.proj = nn.Linear(token_dim, dim)
        self.pos = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x) + self.pos)


class AttentionPoolAdapter(nn.Module):
    """Node embeddings (variable count) -> n_tokens super-node tokens.
    GNN pipeline: multi-seed attention pooling over grain-node embeddings."""

    def __init__(self, node_dim: int, dim: int = 128, n_tokens: int = 8,
                 heads: int = 4):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.proj = nn.Linear(node_dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, node_emb: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """node_emb: (B, N_max, node_dim), key_padding_mask: (B, N_max) True=pad."""
        kv = self.proj(node_emb)
        q = self.seeds.unsqueeze(0).expand(kv.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return self.norm(out + q)
