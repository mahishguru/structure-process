"""FT-Transformer classifier head.

Used as the reranker's complementary auxiliary for the genai and gnn
representations, which have no descriptor-block structure. Wide inputs
(genai) are PCA-compressed first, fitted on train only, so attention stays at
the full batch size.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import ALLOYS, cls_targets, condition_weights

FTT_BATCH = 256
FTT_HEADS = 8
FTT_ATTENTION_BUDGET_BYTES = 1_700_000_000


def _make_ftt(input_dim, output_dim):
    from rtdl_revisiting_models import FTTransformer

    return FTTransformer(
        n_cont_features=input_dim, cat_cardinalities=[],
        n_blocks=2, d_block=128, attention_n_heads=FTT_HEADS,
        attention_dropout=0.2, ffn_d_hidden_multiplier=2.0,
        ffn_dropout=0.1, residual_dropout=0.0,
        d_out=output_dim).to("cuda")


def ftt_micro_batch(n_features: int) -> int:
    """Largest chunk whose attention logits stay inside the memory budget."""
    tokens = n_features + 1
    per_sample = FTT_HEADS * tokens * tokens * 4
    return max(1, min(FTT_BATCH, int(FTT_ATTENTION_BUDGET_BYTES
                                     // per_sample)))


def pca_compress_parts(parts, variance=0.95, cap=128):
    """Reduce a wide representation, fitted on train only. Returns
    (compressed_parts, n_components)."""
    from sklearn.decomposition import PCA

    train = parts[0][0]
    probe = PCA(n_components=min(cap, train.shape[0] - 1, train.shape[1]),
                random_state=0).fit(train)
    n = int(np.searchsorted(
        np.cumsum(probe.explained_variance_ratio_), variance) + 1)
    pca = PCA(n_components=n, random_state=0).fit(train)
    return tuple((pca.transform(x), c) for x, c in parts), n


def fit_ftt_classifier(X_train, train_conditions, balanced=False,
                       epochs=30):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device="cuda")
    classes = torch.as_tensor(
        cls_targets(train_conditions), dtype=torch.long, device="cuda")
    weights = torch.as_tensor(
        condition_weights(train_conditions) if balanced
        else np.ones(len(train_conditions)),
        dtype=torch.float32, device="cuda")
    model = _make_ftt(X_train.shape[1], len(ALLOYS))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)
    micro = ftt_micro_batch(X_train.shape[1])
    for _ in range(epochs):
        permutation = torch.randperm(len(X_tensor), device="cuda")
        for start in range(0, len(X_tensor), FTT_BATCH):
            batch = permutation[start:start + FTT_BATCH]
            optimizer.zero_grad()
            for offset in range(0, len(batch), micro):
                indices = batch[offset:offset + micro]
                output = model(X_tensor[indices], None)
                per_sample = functional.cross_entropy(
                    output, classes[indices], reduction="none")
                loss = (per_sample * weights[indices]).sum() / len(batch)
                loss.backward()
            optimizer.step()
    return model.eval()


def predict_ftt_classifier(model, features) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.as_tensor(features, dtype=torch.float32, device="cuda")
    micro = ftt_micro_batch(features.shape[1])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensor), micro):
            outputs.append(model(tensor[start:start + micro], None))
    output = torch.cat(outputs)
    return functional.softmax(output, dim=1).cpu().numpy()
