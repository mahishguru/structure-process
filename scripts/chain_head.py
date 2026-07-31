#!/usr/bin/env python
"""Autoregressive CatBoost classifier chain over element levels.

Type-correct head for the inverse structure->recipe task:
  p(y|x) = prod_e p(level_e | x, levels_<e) * p(T,v | x, levels)
Element amounts are categorical over the observed training lattice (level 0 =
absent, so no separate hurdle); the chain captures inter-element constraints;
process parameters are regressed with the decoded composition appended.

Unlike a direct 14-way alloy classifier, the chain assigns probability to
UNSEEN combinations of seen levels, so it can rank a held-out alloy in LOAO.

Decodes:
  greedy      argmax level per element, chain-fed
  alloyscore  exact chain probability of each candidate alloy's level vector
              (teacher-forced scoring); composition = argmax alloy nominal
  fuse        0.5 * chain alloy probs + 0.5 * kNN vote distribution

Usage: tabular chain over conventional features, LOCO and LOAO.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBRegressor

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


def wape_present(yt, yp):
    vals = []
    for j in range(8):
        m = yt[:, j] > 0
        if m.sum():
            vals.append(np.abs(yt[m, j] - yp[m, j]).sum() / yt[m, j].sum())
    return float(np.mean(vals))


def fit_chain(Xtr, Wtr):
    """One CatBoost per element, teacher-forced on true previous levels.
    Returns list of (levels_e, model_or_None)."""
    chain = []
    prev = np.empty((len(Xtr), 0))
    for e in range(8):
        w = Wtr[:, e]
        levels = np.unique(np.round(w, 4))
        if len(levels) == 1:
            chain.append((levels, None))
        else:
            y = np.searchsorted(levels, np.round(w, 4))
            m = CatBoostClassifier(loss_function="MultiClass",
                                   iterations=300, depth=6,
                                   learning_rate=0.1, verbose=0,
                                   task_type="GPU", devices="0")
            m.fit(np.hstack([Xtr, prev]), y)
            chain.append((levels, m))
        prev = np.hstack([prev, w[:, None]])
    return chain


def chain_probs(chain, X, prev_fixed):
    """Per-element level probabilities with previous levels forced to
    prev_fixed (n_samples x 8). Returns list of (levels, prob (N, L))."""
    out = []
    prev = np.empty((len(X), 0))
    for e, (levels, m) in enumerate(chain):
        if m is None:
            p = np.ones((len(X), 1))
        else:
            p = np.zeros((len(X), len(levels)))
            raw = np.asarray(m.predict_proba(np.hstack([X, prev])))
            p[:, [int(c) for c in m.classes_]] = raw
        out.append((levels, p))
        prev = np.hstack([prev, prev_fixed[:, e][:, None]])
    return out


def greedy_decode(chain, X):
    prev = np.empty((len(X), 0))
    W = np.zeros((len(X), 8))
    for e, (levels, m) in enumerate(chain):
        if m is None:
            w = np.full(len(X), levels[0])
        else:
            raw = np.asarray(m.predict_proba(np.hstack([X, prev])))
            p = np.zeros((len(X), len(levels)))
            p[:, [int(c) for c in m.classes_]] = raw
            w = levels[p.argmax(1)]
        W[:, e] = w
        prev = np.hstack([prev, w[:, None]])
    return W


def score_alloys(chain, X, candidates):
    """log p(alloy) under the chain for each candidate index, exact
    teacher-forced scoring. Returns (N, n_cand)."""
    logp = np.zeros((len(X), len(candidates)))
    for ci, ai in enumerate(candidates):
        target = NOMINAL[ai]
        fixed = np.tile(target, (len(X), 1))
        probs = chain_probs(chain, X, fixed)
        for e, (levels, p) in enumerate(probs):
            t = np.round(target[e], 4)
            j = np.searchsorted(levels, t)
            if j >= len(levels) or not np.isclose(levels[j], t):
                logp[:, ci] += np.log(1e-6)      # level unseen in training
            else:
                logp[:, ci] += np.log(np.clip(p[:, j], 1e-6, 1))
    return logp


def knn_vote(Xtr, Xte, ctr, k=5, temp=0.05):
    ycls = np.array([ALLOYS.index(a) for a in ALLOY.loc[ctr]])
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


def eval_fold(d: str) -> dict:
    split = Path(d).name
    Xtr, Xte, ctr, cte = load_fold(d)
    Wtr = LB.loc[ctr].to_numpy(float)[:, :8]
    chain = fit_chain(Xtr, Wtr)

    held = split[len("loao_"):] if split.startswith("loao_") else None
    cands = list(range(len(ALLOYS)))      # chain CAN score the held-out alloy

    W_greedy = greedy_decode(chain, Xte)
    logp = score_alloys(chain, Xte, cands)

    # process params: regressor on features + greedy composition
    Ptr = LB.loc[ctr].to_numpy(float)[:, 8:]
    Xtr_aug = np.hstack([Xtr, Wtr])
    Xte_aug = np.hstack([Xte, W_greedy])
    proc = np.zeros((len(Xte), 2))
    for j in range(2):
        r = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                         tree_method="hist", device="cuda", n_jobs=-1,
                         verbosity=0)
        r.fit(Xtr_aug, Ptr[:, j])
        proc[:, j] = r.predict(Xte_aug)

    pk = knn_vote(Xtr, Xte, ctr)

    conds = sorted(set(cte))
    yt = np.stack([LB.loc[c].to_numpy(float) for c in conds])
    yti = np.array([ALLOYS.index(ALLOY.loc[c]) for c in conds])
    proc_c = np.stack([proc[cte == c].mean(0) for c in conds])
    res = {}
    variants = {
        "greedy": np.stack([W_greedy[cte == c].mean(0) for c in conds]),
    }
    # alloy-scored decodes (mean log-prob over the condition's images)
    lp_c = np.stack([logp[cte == c].mean(0) for c in conds])
    p_chain = np.exp(lp_c - lp_c.max(1, keepdims=True))
    p_chain /= p_chain.sum(1, keepdims=True)
    pk_c = np.stack([pk[cte == c].mean(0) for c in conds])
    for name, P in [("alloyscore", p_chain),
                    ("fuse", 0.5 * p_chain + 0.5 * pk_c)]:
        order = np.argsort(-P, 1)
        variants[name] = NOMINAL[order[:, 0]]
        res[f"{name}_top1"] = float(np.mean(order[:, 0] == yti))
        res[f"{name}_top3"] = float(np.mean(
            [yti[i] in order[i, :3] for i in range(len(conds))]))
        res[f"{name}_class_nll"] = float(-np.log(np.clip(
            P[np.arange(len(conds)), yti], 1e-12, 1)).mean())
    for name, comp in variants.items():
        res[f"{name}_wape"] = wape_present(yt, comp)
        res[f"{name}_mae"] = float(np.abs(yt[:, :8] - comp).mean())
    res["T_ext_wape"] = float(np.abs(yt[:, 8] - proc_c[:, 8 - 8]).sum()
                              / np.abs(yt[:, 8]).sum())
    res["split"] = split
    if held is not None:
        res["held_out_rank_alloyscore"] = float(np.mean(
            np.argsort(-p_chain, 1).argsort(1)[
                np.arange(len(conds)), yti]))
    Path(f"runs/chain_{split}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loco", "loao"], required=True)
    args = ap.parse_args()
    pat = ("data/conventional/loco_fold*" if args.mode == "loco"
           else "data/conventional/loao_*")
    reps = []
    for d in sorted(glob.glob(pat)):
        reps.append(eval_fold(d))
        print(f"  {Path(d).name} done", flush=True)
    keys = [k for k in reps[0] if k != "split"
            and all(k in r for r in reps)]
    summary = {k: float(np.mean([r[k] for r in reps])) for k in keys}
    summary["n_folds"] = len(reps)
    Path(f"runs/summary_chain_{args.mode}.json").write_text(
        json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k:28} {v:.3f}")


if __name__ == "__main__":
    main()
