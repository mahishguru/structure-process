#!/usr/bin/env python
"""Unified head study: the SAME prediction heads applied to all three
representation families (conventional descriptors, genai latents, GNN
embeddings extracted from trained per-fold encoders).

Heads:
  knn     cosine kNN retrieval over the composition lattice (k=5)
  cat     CatBoost 14-way alloy classifier + XGB process regressor
  fuse    0.5*CatBoost probs + 0.5*kNN vote  (previous LOCO winner)
  gp      condition-level Gaussian-process head: GPC over the alloy
          lattice + exact GPR (RBF+White) for T_ext/v_ext with
          predictive variance (NLL, 90% coverage)

Protocol identical to tabular_heads.py: image-level training (except gp,
which is condition-level by design), per-condition aggregation,
present-only element WAPE, element MAE, alloy top-1/3, T_ext WAPE.

Usage:
  .venv/bin/python scripts/unified_heads.py --pipeline conventional
  .venv/bin/python scripts/unified_heads.py --pipeline genai
  .venv/bin/python scripts/unified_heads.py --pipeline gnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from tabular_heads import (ALLOYS, ELEM, LB, NOMINAL, _cls_decode, _proc_xgb,
                           cls_targets, metrics, reg_targets)

FOLDS = [f"loco_fold{i}" for i in range(5)]


# ------------------------------------------------------------ features
def load_conventional(split: str):
    d = f"data/conventional/{split}"
    Xtr = np.load(f"{d}/X_train.npy")
    Xte = np.load(f"{d}/X_test.npy")
    ctr = np.array(json.load(open(f"{d}/conditions_train.json")))
    cte = np.array(json.load(open(f"{d}/conditions_test.json")))
    return Xtr, Xte, ctr, cte


_GENAI = None


def load_genai(split: str):
    global _GENAI
    if _GENAI is None:
        d = np.load("data/genai/latents.npz", allow_pickle=True)
        _GENAI = (d["latents"].reshape(len(d["latents"]), -1),
                  np.array([str(c) for c in d["condition_ids"]]))
    X, cid = _GENAI
    sp = json.loads(Path(f"data/splits/{split}.json").read_text())
    mtr = np.isin(cid, sp["train"])
    mte = np.isin(cid, sp["test"])
    return X[mtr], X[mte], cid[mtr], cid[mte]


def load_gnn(split: str):
    """Extract per-graph embeddings from the trained per-fold encoder."""
    cache = Path(f"data/gnn/embeddings/{split}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["Xtr"], d["Xte"], d["ctr"], d["cte"]
    import torch
    import yaml
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from icme_mg.models.gnn_encoder import GrainGraphEncoder
    from icme_mg.training.datasets import GrainGraphDataset

    cfg = yaml.safe_load(Path("configs/flow_head.yaml").read_text())
    gcfg = yaml.safe_load(Path("configs/gnn.yaml").read_text())["gnn_encoder"]
    enc = GrainGraphEncoder(
        hidden_dim=gcfg["hidden_dim"], num_layers=gcfg["num_layers"],
        heads=gcfg["heads"], pool_tokens=gcfg["pool_tokens"],
        out_dim=cfg["head"]["dim"], dropout=gcfg["dropout"])
    ckpt = torch.load(f"runs/gnn_{split}_s90/best.pt", map_location="cuda",
                      weights_only=False)
    state = {k[len("encoder."):]: v for k, v in ckpt["model"].items()
             if k.startswith("encoder.")}
    enc.load_state_dict(state)
    enc.to("cuda").eval()

    sp = json.loads(Path(f"data/splits/{split}.json").read_text())
    out = {}
    for part in ("train", "test"):
        ds = GrainGraphDataset(Path("data/gnn/synthetic/graphs"),
                               "data/labels/labels.csv", sp[part],
                               scales=[90])
        feats, cids = [], []
        with torch.no_grad():
            for batch in GeoDataLoader(ds, batch_size=64):
                z = enc(batch.to("cuda"))            # (B, T, D)
                feats.append(z.mean(1).cpu().numpy())
                cids.extend(list(batch.condition_id))
        out[part] = (np.concatenate(feats), np.array(cids))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, Xtr=out["train"][0], Xte=out["test"][0],
             ctr=out["train"][1], cte=out["test"][1])
    return out["train"][0], out["test"][0], out["train"][1], out["test"][1]


LOADERS = {"conventional": load_conventional, "genai": load_genai,
           "gnn": load_gnn}


def zscore(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ---------------------------------------------------------------- heads
def knn_probs(Xtr, Xte, ycls, k=5, temp=0.05):
    An = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Bn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    sim = Bn @ An.T
    idx = np.argsort(-sim, 1)[:, :k]
    w = np.take_along_axis(sim, idx, 1)
    w = np.exp((w - w.max(1, keepdims=True)) / temp)
    pk = np.zeros((len(Xte), len(ALLOYS)))
    for i in range(len(Xte)):
        for j, wt in zip(ycls[idx[i]], w[i]):
            pk[i, j] += wt
    return pk / pk.sum(1, keepdims=True)


def cat_probs(Xtr, Xte, ycls):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function="MultiClass", iterations=500,
                           depth=6, learning_rate=0.1, verbose=0,
                           task_type="GPU", devices="0")
    m.fit(Xtr, ycls)
    p = np.zeros((len(Xte), len(ALLOYS)))
    p[:, [int(c) for c in m.classes_]] = np.asarray(m.predict_proba(Xte))
    return p


def run_img_heads(Xtr, Xte, ctr, cte):
    """kNN / CatBoost / fusion — image level, shared process regressor."""
    ycls = cls_targets(ctr)
    proc = _proc_xgb(Xtr, Xte, ctr)
    pk = knn_probs(Xtr, Xte, ycls)
    pc = cat_probs(Xtr, Xte, ycls)
    res = {}
    for name, p in [("knn", pk), ("cat", pc), ("fuse", 0.5 * pc + 0.5 * pk)]:
        res[name] = _cls_decode(p, cte, proc)["argmax"]
        # per-condition class NLL for calibration comparison
        from tabular_heads import ALLOY
        conds = sorted(set(cte))
        prob = np.stack([p[cte == c].mean(0) for c in conds])
        ti = np.array([ALLOYS.index(ALLOY.loc[c]) for c in conds])
        res[name]["class_nll"] = float(
            -np.log(np.maximum(prob[np.arange(len(conds)), ti], 1e-9)).mean())
    return res


def run_gp(Xtr, Xte, ctr, cte):
    """Condition-level GP head: GPC over the lattice + GPR for T/v."""
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import (GaussianProcessClassifier,
                                          GaussianProcessRegressor)
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from tabular_heads import ALLOY

    def pool(X, cid):
        conds = sorted(set(cid))
        return conds, np.stack([X[cid == c].mean(0) for c in conds])

    ktr, Ztr = pool(Xtr, ctr)
    kte, Zte = pool(Xte, cte)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    pca = PCA(n_components=min(32, len(Ztr) - 1, Ztr.shape[1]))
    Ztr, Zte = pca.fit_transform(Ztr), pca.transform(Zte)

    ycls = np.array([ALLOYS.index(ALLOY.loc[c]) for c in ktr])
    kern = ConstantKernel(1.0) * RBF(length_scale=np.sqrt(Ztr.shape[1]))
    gpc = GaussianProcessClassifier(kernel=kern, max_iter_predict=100)
    gpc.fit(Ztr, ycls)
    prob = np.zeros((len(Zte), len(ALLOYS)))
    prob[:, gpc.classes_] = gpc.predict_proba(Zte)

    Y = LB.loc[ktr].to_numpy(float)[:, 8:]
    proc, nll, cov = np.zeros((len(Zte), 2)), [], []
    t_true = LB.loc[kte].to_numpy(float)[:, 8]
    for j in range(2):
        ym, ys = Y[:, j].mean(), Y[:, j].std() + 1e-8
        gpr = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(np.sqrt(Ztr.shape[1]))
            + WhiteKernel(0.1), normalize_y=False, alpha=1e-6)
        gpr.fit(Ztr, (Y[:, j] - ym) / ys)
        m, s = gpr.predict(Zte, return_std=True)
        proc[:, j] = m * ys + ym
        if j == 0:
            sd_p = np.maximum(s * ys, 1e-6)
            z = (t_true - proc[:, 0]) / sd_p
            nll = float(np.mean(0.5 * z**2 + np.log(sd_p)
                                + 0.5 * np.log(2 * np.pi)))
            cov = float(np.mean(np.abs(z) < 1.6449))

    order = np.argsort(-prob, 1)
    yp = np.concatenate([NOMINAL[order[:, 0]], proc], 1)
    res = metrics(kte, yp, top_idx=order)
    ti = np.array([ALLOYS.index(ALLOY.loc[c]) for c in kte])
    res["class_nll"] = float(
        -np.log(np.maximum(prob[np.arange(len(kte)), ti], 1e-9)).mean())
    res["T_nll"] = nll
    res["T_coverage90"] = cov
    return {"gp": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=list(LOADERS))
    ap.add_argument("--heads", nargs="*", default=["img", "gp"])
    args = ap.parse_args()
    per: dict[str, list[dict]] = {}
    for split in FOLDS:
        Xtr, Xte, ctr, cte = LOADERS[args.pipeline](split)
        Xtr, Xte = zscore(Xtr, Xte)
        res = {}
        if "img" in args.heads:
            res.update(run_img_heads(Xtr, Xte, ctr, cte))
        if "gp" in args.heads:
            res.update(run_gp(Xtr, Xte, ctr, cte))
        for k, v in res.items():
            per.setdefault(k, []).append(v)
        print(f"  {split} done", flush=True)
    summary = {}
    for name, reps in per.items():
        keys = set.intersection(*[set(r) for r in reps])
        summary[name] = {k: float(np.mean([r[k] for r in reps]))
                         for k in sorted(keys)}
        line = "  ".join(f"{k} {v:.3f}" for k, v in summary[name].items())
        print(f"{args.pipeline}/{name:6} {line}")
    Path(f"runs/summary_unified_{args.pipeline}_loco.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
