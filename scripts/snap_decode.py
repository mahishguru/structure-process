#!/usr/bin/env python
"""Lattice-snapped decoding of existing prediction dumps (no retraining).

Point estimate = posterior mean of the composition samples, snapped to the
nearest TRAINING alloy's nominal composition (z-scored element space).
Process labels keep the posterior mean. For loao_X splits the candidate set
excludes the held-out alloy X (honest out-of-support decoding).

Writes snap_eval_<split>.json next to each predictions_<split>.npz and prints
per-pipeline summaries.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ELEM = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd"]

labels = pd.read_csv("data/labels/labels.csv").set_index("condition_id")
ALLOYS = sorted(labels["alloy"].unique())
NOMINAL = np.stack([
    labels[labels["alloy"] == a].iloc[0][ELEM].to_numpy(float)
    for a in ALLOYS])
MU, SD = NOMINAL.mean(0), NOMINAL.std(0) + 1e-8


def snap(y_pred: np.ndarray, cand: list[int]) -> tuple[np.ndarray, np.ndarray]:
    nom = NOMINAL[cand]
    d = np.linalg.norm(((y_pred[:, :8] - MU) / SD)[:, None]
                       - ((nom - MU) / SD)[None], axis=2)
    out = y_pred.copy()
    out[:, :8] = nom[d.argmin(1)]
    return out, d


def f1_macro(t: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for e in range(8):
        tp = float((t[:, e] & p[:, e]).sum())
        prec = tp / max(p[:, e].sum(), 1)
        rec = tp / max(t[:, e].sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float(np.mean(f1s))


def eval_run(run_dir: str) -> dict | None:
    preds = glob.glob(f"{run_dir}/predictions_*.npz")
    if not preds:
        return None
    split = re.match(r".*predictions_(.+)\.npz", preds[0]).group(1)
    z = np.load(preds[0], allow_pickle=True)
    yt, cids = z["y_true"], z["condition_ids"]
    pm = z["samples"].mean(1)                       # posterior mean, (B, 10)

    if split.startswith("loao_"):
        held = split[len("loao_"):]
        cand = [i for i, a in enumerate(ALLOYS) if a != held]
    else:
        cand = list(range(len(ALLOYS)))
    ysnap, dist = snap(pm, cand)

    true_alloy = [labels.loc[c, "alloy"] for c in cids]
    order = dist.argsort(1)
    cand_names = [ALLOYS[j] for j in cand]
    top1 = float(np.mean([true_alloy[i] == cand_names[order[i, 0]]
                          for i in range(len(true_alloy))]))
    top3 = float(np.mean([true_alloy[i] in [cand_names[j]
                                            for j in order[i, :3]]
                          for i in range(len(true_alloy))]))
    mae = {e: float(np.abs(yt[:, j] - ysnap[:, j]).mean())
           for j, e in enumerate(ELEM)}
    mae["T_ext"] = float(np.abs(yt[:, 8] - pm[:, 8]).mean())
    mae["v_ext"] = float(np.abs(yt[:, 9] - pm[:, 9]).mean())
    rep = {
        "split": split, "decode": "posterior_mean+snap_to_alloy",
        "mae_elements_macro": float(np.mean([mae[e] for e in ELEM])),
        "presence_f1_macro": f1_macro(yt[:, :8] > 0, ysnap[:, :8] > 0),
        "alloy_top1": top1, "alloy_top3": top3, "per_label_mae": mae,
    }
    Path(f"{run_dir}/snap_eval_{split}.json").write_text(
        json.dumps(rep, indent=2))
    return rep


def main() -> None:
    groups = {
        ("gnn", "loco"): "runs/gnn_loco_*_s90",
        ("gnn", "loao"): "runs/gnn_loao_*_s90",
        ("conventional", "loco"): "runs/conventional_loco_*",
        ("conventional", "loao"): "runs/conventional_loao_*",
        ("genai", "loco"): "runs/genai_loco_*",
        ("genai", "loao"): "runs/genai_loao_*",
    }
    for (pipe, mode), pat in groups.items():
        reps = [r for d in sorted(glob.glob(pat))
                if Path(d).is_dir() and (r := eval_run(d))]
        if not reps:
            continue
        agg = {k: float(np.mean([r[k] for r in reps]))
               for k in ("mae_elements_macro", "presence_f1_macro",
                         "alloy_top1", "alloy_top3")}
        agg["n_folds"] = len(reps)
        Path(f"runs/summary_snap_{mode}_{pipe}.json").write_text(
            json.dumps(agg, indent=2))
        print(f"{pipe:13} {mode}  elMAE {agg['mae_elements_macro']:.3f}  "
              f"presF1 {agg['presence_f1_macro']:.3f}  "
              f"top1 {agg['alloy_top1']:.2f}  top3 {agg['alloy_top3']:.2f}  "
              f"(n={len(reps)})")


if __name__ == "__main__":
    main()
