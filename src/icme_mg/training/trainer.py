"""Training loop for {adapter (+ optional GNN encoder)} + probabilistic head.

Recipe (strategy/01 section 5): AdamW 3e-4, cosine schedule with 5% warmup,
batch 64, grad clip 1.0, EMA 0.999, tolerance dequantization each batch,
aux alloy-class CE (weight 0.3, annealed to 0 by 50% of training), early stop
on per-condition val joint NLL.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import BalancedConditionSampler
from icme_mg.models.label_space import LabelNormalizer, dequantize


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def cosine_warmup(step: int, total: int, warmup_frac: float = 0.05) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t))


class Trainer:
    """model = nn.ModuleDict({'adapter': ..., 'head': ...}) with optional
    'encoder' (GNN). batch_to_cond maps a dataloader batch to
    (cond_tokens, y, present, class_idx)."""

    def __init__(self, model: nn.ModuleDict, normalizer: LabelNormalizer,
                 batch_to_cond, out_dir: str | Path, lr: float = 3e-4,
                 weight_decay: float = 1e-4, epochs: int = 200,
                 aux_weight: float = 0.3, aux_anneal_frac: float = 0.5,
                 ema_decay: float = 0.999, grad_clip: float = 1.0,
                 patience: int = 25, device: str = "cpu", seed: int = 0):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.batch_to_cond = batch_to_cond
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epochs, self.device = epochs, device
        self.aux_weight, self.aux_anneal_frac = aux_weight, aux_anneal_frac
        self.grad_clip, self.patience = grad_clip, patience
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.ema = EMA(model, ema_decay)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def _loss(self, batch, progress: float) -> torch.Tensor:
        cond, y, present, class_idx = self.batch_to_cond(batch, self.model,
                                                         self.device)
        y_dq = dequantize(y, generator=self.gen if y.device.type == "cpu"
                          else None)
        z, log_det = self.normalizer.normalize(y_dq)
        out = self.model["head"](z, present, log_det, cond)
        loss = out["nll"].mean()
        aux_w = self.aux_weight * max(0.0, 1 - progress / self.aux_anneal_frac)
        if aux_w > 0:
            loss = loss + aux_w * F.cross_entropy(out["aux_class_logits"],
                                                  class_idx)
        return loss

    @torch.no_grad()
    def _val_nll(self, loader) -> float:
        """Per-condition-averaged joint NLL (physical units)."""
        self.model.eval()
        per_cond: dict[str, list[float]] = {}
        for batch in loader:
            cond, y, present, _ = self.batch_to_cond(batch, self.model,
                                                     self.device)
            z, log_det = self.normalizer.normalize(y)
            out = self.model["head"](z, present, log_det, cond)
            cids = (batch["condition_id"] if isinstance(batch, dict)
                    else batch.condition_id)
            if isinstance(cids, str):
                cids = [cids]
            for c, v in zip(cids, out["nll"].tolist()):
                per_cond.setdefault(c, []).append(v)
        self.model.train()
        return sum(sum(v) / len(v) for v in per_cond.values()) / len(per_cond)

    def fit(self, train_loader, val_loader,
            sampler: BalancedConditionSampler | None = None) -> dict:
        import time
        total_steps = self.epochs * max(1, len(train_loader))
        base_lrs = [g["lr"] for g in self.opt.param_groups]
        best_nll, best_epoch, step = float("inf"), -1, 0
        history = []
        self.model.train()
        for epoch in range(self.epochs):
            t0 = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in train_loader:
                scale = cosine_warmup(step, total_steps)
                for g, lr0 in zip(self.opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale
                loss = self._loss(batch, progress=step / total_steps)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_clip)
                self.opt.step()
                self.ema.update(self.model)
                step += 1
            eval_model = copy.deepcopy(self.model)
            self.ema.copy_to(eval_model)
            eval_model.eval()
            saved, self.model = self.model, eval_model
            val_nll = self._val_nll(val_loader)
            self.model = saved
            self.model.train()
            history.append({"epoch": epoch, "val_nll": val_nll,
                            "train_loss": float(loss.detach())})
            print(f"epoch {epoch:3d}  train_loss {float(loss.detach()):8.4f}  "
                  f"val_nll {val_nll:8.4f}  best {min(best_nll, val_nll):8.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if val_nll < best_nll - 1e-4:
                best_nll, best_epoch = val_nll, epoch
                torch.save({"model": self.ema.shadow,
                            "epoch": epoch, "val_nll": val_nll},
                           self.out_dir / "best.pt")
            elif epoch - best_epoch >= self.patience:
                break
        torch.save(history, self.out_dir / "history.pt")
        return {"best_val_nll": best_nll, "best_epoch": best_epoch,
                "epochs_run": len(history)}


def dict_batch_to_cond(batch, model, device):
    """batch_to_cond for ConventionalDataset / GenAILatentDataset."""
    cond = model["adapter"](batch["x"].to(device))
    return (cond, batch["y"].to(device), batch["present"].to(device),
            batch["class_idx"].to(device))


def graph_batch_to_cond(batch, model, device):
    """batch_to_cond for GrainGraphDataset (torch_geometric Batch)."""
    batch = batch.to(device)
    cond = model["encoder"](batch)
    return cond, batch.y, batch.present, batch.class_idx.view(-1)
