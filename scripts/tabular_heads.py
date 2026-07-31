#!/usr/bin/env python
"""Tabular model bake-off for the inverse structure->recipe task (LOCO).

Protocol (matches knn_retrieval.py): image-level training on cached
conventional features, per-condition aggregation of predictions (mean of
regressed values / mean of class probabilities), present-only element WAPE,
element MAE, alloy top-1/top-3, T_ext WAPE.

Variants:
  xgb_reg   XGBoost regression, one model per target
  cat_reg   CatBoost MultiRMSE multi-output regression
  xgb_cls   XGBoost 14-way alloy classifier -> composition from nominals
            (argmax and probability-weighted mixture) + xgb_reg for T/v
  cat_cls   CatBoost MultiClass alloy classifier, same decode
  ftt       FT-Transformer (rtdl) classifier + regressor
  unc       distributional T_ext: NGBoost Normal and CatBoost
            RMSEWithUncertainty; report NLL and 90% coverage
    fuse      CatBoost + kNN probability fusion
    fuse_balanced
                        fusion with equal total training weight per condition
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ELEM = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd", "Y"]
labels = pd.read_csv("data/labels/labels.csv").set_index("condition_id")
LB = labels[ELEM + ["T_ext", "v_ext"]]
ALLOY = labels["alloy"]
ALLOYS = sorted(labels["alloy"].unique())
NOMINAL = np.stack([labels[labels["alloy"] == a].iloc[0][ELEM]
                    .to_numpy(float) for a in ALLOYS])


def load_fold(d: str):
    Xtr = np.load(f"{d}/X_train.npy")
    Xte = np.load(f"{d}/X_test.npy")
    ctr = np.array(json.load(open(f"{d}/conditions_train.json")))
    cte = np.array(json.load(open(f"{d}/conditions_test.json")))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd, ctr, cte


def agg_conditions(pred_img: np.ndarray, cte: np.ndarray):
    conds = sorted(set(cte))
    return conds, np.stack([pred_img[cte == c].mean(0) for c in conds])


def metrics(conds, yp: np.ndarray, top_idx: np.ndarray | None = None):
    yt = np.stack([LB.loc[c].to_numpy(float) for c in conds])
    wape_vals = []
    for j in range(8):
        m = yt[:, j] > 0
        if m.sum():
            wape_vals.append(np.abs(yt[m, j] - yp[m, j]).sum()
                             / yt[m, j].sum())
    out = {
        "element_wape_present": float(np.mean(wape_vals)),
        "mae_elements_macro": float(np.abs(yt[:, :8] - yp[:, :8]).mean()),
        "T_ext_wape": float(np.abs(yt[:, 8] - yp[:, 8]).sum()
                            / np.abs(yt[:, 8]).sum()),
    }
    if top_idx is None:        # nearest-nominal ranking for pure regressors
        mu, sd = NOMINAL.mean(0), NOMINAL.std(0) + 1e-8
        dist = np.linalg.norm(((yp[:, :8] - mu) / sd)[:, None]
                              - ((NOMINAL - mu) / sd)[None], axis=2)
        top_idx = dist.argsort(1)
    ta = [ALLOY.loc[c] for c in conds]
    out["alloy_top1"] = float(np.mean(
        [ta[i] == ALLOYS[top_idx[i, 0]] for i in range(len(ta))]))
    out["alloy_top3"] = float(np.mean(
        [ta[i] in [ALLOYS[j] for j in top_idx[i, :3]] for i in range(len(ta))]))
    return out


def reg_targets(ctr):
    return LB.loc[ctr].to_numpy(float)


def cls_targets(ctr):
    return np.array([ALLOYS.index(a) for a in ALLOY.loc[ctr]])


def condition_weights(condition_ids):
    """Per-image weights giving every processing condition equal total mass."""
    condition_ids = np.asarray(condition_ids)
    _, inverse, counts = np.unique(
        condition_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def run_xgb_reg(Xtr, Xte, ctr, cte):
    from xgboost import XGBRegressor
    Y = reg_targets(ctr)
    P = np.zeros((len(Xte), 10))
    for j in range(10):
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                         tree_method="hist", device="cuda", n_jobs=-1,
                         verbosity=0)
        m.fit(Xtr, Y[:, j])
        P[:, j] = m.predict(Xte)
    conds, yp = agg_conditions(P, cte)
    return {"xgb_reg": metrics(conds, yp)}


def run_cat_reg(Xtr, Xte, ctr, cte):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function="MultiRMSE", iterations=500,
                          depth=6, learning_rate=0.1, verbose=0,
                          boosting_type="Plain",
                          task_type="GPU", devices="0")
    m.fit(Xtr, reg_targets(ctr))
    conds, yp = agg_conditions(np.asarray(m.predict(Xte)), cte)
    return {"cat_reg": metrics(conds, yp)}


def _cls_decode(prob_img, cte, proc_pred_img):
    conds = sorted(set(cte))
    prob = np.stack([prob_img[cte == c].mean(0) for c in conds])
    proc = np.stack([proc_pred_img[cte == c].mean(0) for c in conds])
    order = np.argsort(-prob, axis=1)
    res = {}
    for name, comp in [("argmax", NOMINAL[order[:, 0]]),
                       ("soft", prob @ NOMINAL)]:
        yp = np.concatenate([comp, proc], 1)
        res[name] = metrics(conds, yp, top_idx=order)
    return res


def _proc_xgb(Xtr, Xte, ctr, sample_weight=None):
    from xgboost import XGBRegressor
    Y = reg_targets(ctr)[:, 8:]
    P = np.zeros((len(Xte), 2))
    for j in range(2):
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                         tree_method="hist", device="cuda", n_jobs=-1,
                         verbosity=0)
        m.fit(Xtr, Y[:, j], sample_weight=sample_weight)
        P[:, j] = m.predict(Xte)
    return P


def run_xgb_cls(Xtr, Xte, ctr, cte):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      tree_method="hist", device="cuda", n_jobs=-1,
                      verbosity=0)
    m.fit(Xtr, cls_targets(ctr))
    prob = m.predict_proba(Xte)
    full = np.zeros((len(Xte), len(ALLOYS)))
    full[:, m.classes_] = prob
    res = _cls_decode(full, cte, _proc_xgb(Xtr, Xte, ctr))
    return {f"xgb_cls_{k}": v for k, v in res.items()}


def run_cat_cls(Xtr, Xte, ctr, cte):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function="MultiClass", iterations=500,
                           depth=6, learning_rate=0.1, verbose=0,
                           task_type="GPU", devices="0")
    m.fit(Xtr, cls_targets(ctr))
    prob = np.asarray(m.predict_proba(Xte))
    cls = [int(c) for c in m.classes_]
    full = np.zeros((len(Xte), len(ALLOYS)))
    full[:, cls] = prob
    res = _cls_decode(full, cte, _proc_xgb(Xtr, Xte, ctr))
    return {f"cat_cls_{k}": v for k, v in res.items()}


def run_ftt(Xtr, Xte, ctr, cte):
    import torch
    import torch.nn.functional as F
    from rtdl_revisiting_models import FTTransformer
    dev = "cuda"
    ycls = torch.as_tensor(cls_targets(ctr), device=dev)
    yreg = torch.as_tensor(reg_targets(ctr)[:, 8:], dtype=torch.float32,
                           device=dev)
    yreg_mu, yreg_sd = yreg.mean(0), yreg.std(0) + 1e-8
    yregz = (yreg - yreg_mu) / yreg_sd
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    kw = dict(n_cont_features=Xtr.shape[1], cat_cardinalities=[],
              n_blocks=2, d_block=128, attention_n_heads=8,
              attention_dropout=0.2, ffn_d_hidden_multiplier=2.0,
              ffn_dropout=0.1, residual_dropout=0.0)
    model = FTTransformer(d_out=len(ALLOYS) + 2, **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n = len(Xtr_t)
    for epoch in range(30):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            out = model(Xtr_t[idx], None)
            loss = (F.cross_entropy(out[:, :len(ALLOYS)], ycls[idx])
                    + F.mse_loss(out[:, len(ALLOYS):], yregz[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        outs = torch.cat([model(Xte_t[i:i + 1024], None)
                          for i in range(0, len(Xte_t), 1024)])
    prob = F.softmax(outs[:, :len(ALLOYS)], 1).cpu().numpy()
    proc = (outs[:, len(ALLOYS):] * yreg_sd + yreg_mu).cpu().numpy()
    res = _cls_decode(prob, cte, proc)
    return {f"ftt_{k}": v for k, v in res.items()}


def run_unc(Xtr, Xte, ctr, cte):
    """Distributional T_ext: per-condition NLL and 90% coverage."""
    out = {}
    conds = sorted(set(cte))
    t_true = np.array([LB.loc[c, "T_ext"] for c in conds])

    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function="RMSEWithUncertainty",
                          iterations=500, depth=6, learning_rate=0.1,
                          verbose=0)
    m.fit(Xtr, reg_targets(ctr)[:, 8])
    p = np.asarray(m.predict(Xte))          # (N, 2): mean, var
    mu = np.array([p[cte == c, 0].mean() for c in conds])
    var = np.array([p[cte == c, 1].mean() + p[cte == c, 0].var()
                    for c in conds])
    sd = np.sqrt(np.maximum(var, 1e-6))
    z = (t_true - mu) / sd
    out["cat_unc_T"] = {
        "nll": float(np.mean(0.5 * z**2 + np.log(sd)
                             + 0.5 * np.log(2 * np.pi))),
        "coverage90": float(np.mean(np.abs(z) < 1.6449)),
        "T_ext_wape": float(np.abs(t_true - mu).sum() / np.abs(t_true).sum()),
    }

    from ngboost import NGBRegressor
    sub = np.random.RandomState(0).choice(len(Xtr), min(1500, len(Xtr)),
                                          replace=False)
    ng = NGBRegressor(n_estimators=200, verbose=False)
    ng.fit(Xtr[sub], reg_targets(ctr)[sub, 8])
    d = ng.pred_dist(Xte)
    mu_i, sd_i = d.params["loc"], d.params["scale"]
    mu = np.array([mu_i[cte == c].mean() for c in conds])
    sd = np.array([np.sqrt((sd_i[cte == c]**2).mean()
                           + mu_i[cte == c].var()) for c in conds])
    z = (t_true - mu) / sd
    out["ngboost_T"] = {
        "nll": float(np.mean(0.5 * z**2 + np.log(sd)
                             + 0.5 * np.log(2 * np.pi))),
        "coverage90": float(np.mean(np.abs(z) < 1.6449)),
        "T_ext_wape": float(np.abs(t_true - mu).sum() / np.abs(t_true).sum()),
    }
    return out


def run_fuse(Xtr, Xte, ctr, cte):
    """CatBoost classifier probs fused 50/50 with kNN vote distribution."""
    return _run_fuse(Xtr, Xte, ctr, cte, balance_conditions=False)


def run_fuse_balanced(Xtr, Xte, ctr, cte):
    """Fusion with equal total training weight per processing condition."""
    return _run_fuse(Xtr, Xte, ctr, cte, balance_conditions=True)


def _run_fuse(Xtr, Xte, ctr, cte, balance_conditions):
    from catboost import CatBoostClassifier
    ycls = cls_targets(ctr)
    sample_weight = condition_weights(ctr) if balance_conditions else None
    m = CatBoostClassifier(loss_function="MultiClass", iterations=500,
                           depth=6, learning_rate=0.1, verbose=0,
                           task_type="GPU", devices="0")
    m.fit(Xtr, ycls, sample_weight=sample_weight)
    pc = np.zeros((len(Xte), len(ALLOYS)))
    pc[:, [int(c) for c in m.classes_]] = np.asarray(m.predict_proba(Xte))
    An = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Bn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    sim = Bn @ An.T
    idx = np.argsort(-sim, 1)[:, :5]
    w = np.take_along_axis(sim, idx, 1)
    w = np.exp((w - w.max(1, keepdims=True)) / 0.05)
    pk = np.zeros((len(Xte), len(ALLOYS)))
    for i in range(len(Xte)):
        vote_weight = w[i]
        if sample_weight is not None:
            vote_weight = vote_weight * sample_weight[idx[i]]
        for j, wt in zip(ycls[idx[i]], vote_weight):
            pk[i, j] += wt
    pk /= pk.sum(1, keepdims=True)
    proc = _proc_xgb(Xtr, Xte, ctr, sample_weight=sample_weight)
    res = _cls_decode(0.5 * pc + 0.5 * pk, cte, proc)
    prefix = "fuse_balanced" if balance_conditions else "fuse"
    return {f"{prefix}_{k}": v for k, v in res.items()}


RUNNERS = {"xgb_reg": run_xgb_reg, "cat_reg": run_cat_reg,
           "xgb_cls": run_xgb_cls, "cat_cls": run_cat_cls,
           "ftt": run_ftt, "unc": run_unc, "fuse": run_fuse,
           "fuse_balanced": run_fuse_balanced}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(RUNNERS))
    args = ap.parse_args()
    folds = sorted(glob.glob("data/conventional/loco_fold*"))
    per_fold: dict[str, list[dict]] = {}
    for d in folds:
        Xtr, Xte, ctr, cte = load_fold(d)
        for name, rep in RUNNERS[args.variant](Xtr, Xte, ctr, cte).items():
            per_fold.setdefault(name, []).append(rep)
        print(f"  {Path(d).name} done", flush=True)
    summary = {}
    for name, reps in per_fold.items():
        summary[name] = {k: float(np.mean([r[k] for r in reps]))
                         for k in reps[0]}
        s = summary[name]
        line = "  ".join(f"{k} {v:.3f}" for k, v in s.items())
        print(f"{name:16} {line}")
    Path(f"runs/summary_tabular_{args.variant}_loco.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
