"""GNN encoder over grain-adjacency graphs (GATv2 + multi-seed attention pooling
to conditioning tokens for the flow head). Trained jointly with the head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .adapters import AttentionPoolAdapter


class GrainGraphEncoder(nn.Module):
    def __init__(self, node_dim: int = 9, edge_dim: int = 2,
                 hidden_dim: int = 128, num_layers: int = 4, heads: int = 4,
                 pool_tokens: int = 8, out_dim: int = 128,
                 dropout: float = 0.1, node_dropout: float = 0.0,
                 edge_dropout: float = 0.0):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // heads, heads=heads,
                edge_dim=edge_dim, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.act = nn.GELU()
        self.pool = AttentionPoolAdapter(hidden_dim, out_dim, pool_tokens)

    def forward(self, batch) -> torch.Tensor:
        """batch: torch_geometric Batch. Returns (B, pool_tokens, out_dim)."""
        from torch_geometric.utils import dropout_edge, to_dense_batch

        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        if self.training and self.edge_dropout > 0:
            edge_index, edge_mask = dropout_edge(edge_index,
                                                 p=self.edge_dropout)
            edge_attr = edge_attr[edge_mask]
        h = self.input_proj(batch.x)
        if self.training and self.node_dropout > 0:
            keep = (torch.rand(h.shape[0], 1, device=h.device)
                    >= self.node_dropout).float()
            h = h * keep / (1 - self.node_dropout)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.act(norm(conv(h, edge_index, edge_attr)))
        dense, mask = to_dense_batch(h, batch.batch)     # (B, N_max, H), (B, N_max)
        return self.pool(dense, key_padding_mask=~mask)
