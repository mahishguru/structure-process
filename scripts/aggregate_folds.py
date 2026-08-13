#!/usr/bin/env python
"""Aggregate eval_{split}.json files across CV folds into paper metrics.

Produces fold-mean +/- 95% CI (Student t) for every headline metric and
writes a JSON summary plus a LaTeX-ready table snippet.

Usage:
  python scripts/aggregate_folds.py --runs runs/gnn_loco_fold0 runs/gnn_loco_fold1 ... \
      --splits loco_fold0 loco_fold1 ... --out runs/summary_loco.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ELEMENTS = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd"]
PROCESS = ["T_ext", "v_ext"]


def mean_ci(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    m = float(arr.mean())
    if n < 2:
        return {"mean": m, "ci95": 0.0, "n": n}
    half = float(stats.t.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "ci95": half, "n": n}


def collect(reports: list[dict]) -> dict:
    out: dict = {}
    out["joint_nll"] = mean_ci([r["joint_nll"] for r in reports])
    out["presence_f1_macro"] = mean_ci(
        [r["presence_f1"]["macro"] for r in reports])
    out["quantile_ece_macro"] = mean_ci(
        [r["quantile_ece"]["macro"] for r in reports])
    out["coverage_90_macro"] = mean_ci(
        [float(np.mean(list(r["coverage_90"].values()))) for r in reports])
    out["alloy_top1"] = mean_ci([r["alloy_topk"]["top1"] for r in reports])
    out["alloy_top3"] = mean_ci([r["alloy_topk"]["top3"] for r in reports])
    out["per_label"] = {}
    for lab in ELEMENTS + PROCESS:
        r2_vals = [r["per_label"][lab]["r2"] for r in reports
                   if "r2" in r["per_label"][lab]
                   and np.isfinite(r["per_label"][lab]["r2"])]
        out["per_label"][lab] = {
            "mae": mean_ci([r["per_label"][lab]["mae"] for r in reports]),
            # r2 is undefined when the held-out labels are constant (LOAO);
            # aggregate only over folds where it exists and is finite.
            "r2": mean_ci(r2_vals) if r2_vals else None,
        }
    out["mae_elements_macro"] = mean_ci(
        [float(np.mean([r["per_label"][e]["mae"] for e in ELEMENTS]))
         for r in reports])
    out["n_folds"] = len(reports)
    return out


def latex_row(name: str, s: dict) -> str:
    def f(k, digits=3):
        v = s[k]
        return f"{v['mean']:.{digits}f} $\\pm$ {v['ci95']:.{digits}f}"
    return (f"{name} & {f('joint_nll')} & {f('presence_f1_macro')} & "
            f"{f('alloy_top1', 2)} & {f('alloy_top3', 2)} & "
            f"{f('quantile_ece_macro')} & {f('coverage_90_macro', 2)} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories, one per fold")
    ap.add_argument("--splits", nargs="+", required=True,
                    help="split name per run dir (same order)")
    ap.add_argument("--name", default="gnn",
                    help="row label for the LaTeX table")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.runs) == len(args.splits)

    reports = []
    for run, split in zip(args.runs, args.splits):
        p = Path(run) / f"eval_{split}.json"
        reports.append(json.loads(p.read_text()))

    summary = collect(reports)
    summary["runs"] = args.runs
    summary["splits"] = args.splits
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_label", "runs", "splits")}, indent=2))
    print("\nLaTeX row:")
    print(latex_row(args.name, summary))


if __name__ == "__main__":
    main()
