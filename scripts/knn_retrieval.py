#!/usr/bin/env python
"""kNN retrieval decoder over cached conventional features.

In the scarce-data regime (14 alloys, ~100 conditions) nonparametric retrieval
beats the amortized flow heads on point prediction. For each test image the
top-k training images are retrieved by cosine similarity in z-scored feature
space; per condition, neighbor labels are pooled over all its images with
temperature-sharpened similarity weights. Elements and process labels are the
weighted label average (winner over alloy-vote/RF/genai-latent variants).

Writes runs/knn_<split>.json per fold and runs/summary_knn_{loco,loao}.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ELEM = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd"]
K = 5
TEMP = 0.05

labels = pd.read_csv("data/labels/labels.csv").set_index("condition_id")
LB = labels[ELEM + ["T_ext", "v_ext"]]
ALLOY = labels["alloy"]


def eval_fold(feat_dir: str) -> dict:
    split = Path(feat_dir).name
    Xtr = np.load(f"{feat_dir}/X_train.npy")
    Xte = np.load(f"{feat_dir}/X_test.npy")
    ctr = np.array(json.load(open(f"{feat_dir}/conditions_train.json")))
    cte = np.array(json.load(open(f"{feat_dir}/conditions_test.json")))

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    sim = B @ A.T

    Ytr = LB.loc[ctr].to_numpy(float)
    atr = ALLOY.loc[ctr].to_numpy()
    conds = sorted(set(cte))
    y_true, y_pred, top1, top3 = [], [], [], []
    for c in conds:
        rows = sim[cte == c]
        idx = np.argsort(-rows, axis=1)[:, :K]
        w = np.take_along_axis(rows, idx, 1)
        w = np.exp((w - w.max()) / TEMP)
        y_pred.append((Ytr[idx] * w[..., None]).sum((0, 1)) / w.sum())
        y_true.append(LB.loc[c].to_numpy(float))
        votes: dict[str, float] = {}
        for a_row, w_row in zip(atr[idx], w):
            for a, wt in zip(a_row, w_row):
                votes[a] = votes.get(a, 0.0) + float(wt)
        rank = sorted(votes, key=votes.get, reverse=True)
        top1.append(ALLOY.loc[c] == rank[0])
        top3.append(ALLOY.loc[c] in rank[:3])
    yt, yp = np.array(y_true), np.array(y_pred)

    wape_vals = []
    per_label = {}
    for j, e in enumerate(ELEM):
        m = yt[:, j] > 0
        per_label[e] = {"mae": float(np.abs(yt[:, j] - yp[:, j]).mean())}
        if m.sum():
            v = float(np.abs(yt[m, j] - yp[m, j]).sum() / yt[m, j].sum())
            per_label[e]["wape_present"] = v
            wape_vals.append(v)
    for j, e in [(8, "T_ext"), (9, "v_ext")]:
        per_label[e] = {
            "mae": float(np.abs(yt[:, j] - yp[:, j]).mean()),
            "wape": float(np.abs(yt[:, j] - yp[:, j]).sum()
                          / np.abs(yt[:, j]).sum())}
    rep = {
        "split": split, "k": K, "temp": TEMP,
        "element_wape_present": float(np.mean(wape_vals)),
        "mae_elements_macro": float(np.abs(yt[:, :8] - yp[:, :8]).mean()),
        "alloy_top1": float(np.mean(top1)), "alloy_top3": float(np.mean(top3)),
        "per_label": per_label,
    }
    Path(f"runs/knn_{split}.json").write_text(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    for mode, pat in [("loco", "data/conventional/loco_fold*"),
                      ("loao", "data/conventional/loao_*")]:
        reps = [eval_fold(d) for d in sorted(glob.glob(pat))]
        agg = {k: float(np.mean([r[k] for r in reps]))
               for k in ("element_wape_present", "mae_elements_macro",
                         "alloy_top1", "alloy_top3")}
        agg["T_ext_wape"] = float(np.mean(
            [r["per_label"]["T_ext"]["wape"] for r in reps]))
        agg["n_folds"] = len(reps)
        Path(f"runs/summary_knn_{mode}.json").write_text(
            json.dumps(agg, indent=2))
        print(f"{mode}: elWAPE {agg['element_wape_present']*100:.1f}%  "
              f"elMAE {agg['mae_elements_macro']:.3f}  "
              f"top1 {agg['alloy_top1']:.2f}  top3 {agg['alloy_top3']:.2f}  "
              f"T_WAPE {agg['T_ext_wape']*100:.1f}%  (n={len(reps)})")


if __name__ == "__main__":
    main()
