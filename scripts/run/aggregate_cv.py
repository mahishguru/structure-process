#!/usr/bin/env python
"""Aggregate 5-fold CV reports into the paper tables.

Reads runs/cv/{fold}/{pipeline}/report.json and writes:
  results/tables/composition_heads_cv.csv   per-fold rows + mean/std summary
  results/tables/process_heads_cv.csv       per-fold rows + mean/std summary
  results/tables/cv_selection_audit.csv     reranker weights, grid-GP decoder,
                                            tuned hyperparameters per fold

Usage: python scripts/run/aggregate_cv.py [--runs-root runs/cv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COMPOSITION_SCALAR = (
    "element_wape_present", "element_wape_all", "element_mae_macro",
    "macro_false_positive_rate", "alloy_top1", "alloy_top3", "class_nll")
PROCESS_TARGETS = ("T_ext", "v_ext", "log_v_ext")
PROCESS_SCALAR = ("mae", "mape", "wape", "r2", "rmse", "fold_error",
                 "nll", "coverage90")


def collect(runs_root: Path):
    composition_rows, process_rows, audit_rows = [], [], []
    for report_path in sorted(runs_root.glob("*/*/report.json")):
        report = json.loads(report_path.read_text())
        fold, pipeline = report["split"], report["pipeline"]
        for head, metrics in report["composition_heads"].items():
            row = {"fold": fold, "pipeline": pipeline, "head": head}
            row.update({key: metrics.get(key) for key in COMPOSITION_SCALAR})
            composition_rows.append(row)
        for head, targets in report["process_heads"].items():
            for target in PROCESS_TARGETS:
                if target not in targets:
                    continue
                row = {"fold": fold, "pipeline": pipeline, "head": head,
                       "target": target}
                row.update({key: targets[target].get(key)
                            for key in PROCESS_SCALAR})
                process_rows.append(row)
        selection = report.get("selection", {})
        reranker = selection.get("composition", {}).get("reranker", {})
        grid_gp = selection.get("process", {}).get("grid_gp", {})
        audit_rows.append({
            "fold": fold, "pipeline": pipeline,
            "reranker_aux": reranker.get("auxiliary"),
            "reranker_owner_weight": reranker.get("selected_owner_weight"),
            "reranker_inner_cv": reranker.get("inner_cv_protocol"),
            "grid_gp_decoder": json.dumps(
                grid_gp.get("selected_velocity_decoder")),
            "knn_params": json.dumps(
                selection.get("composition", {}).get("knn_params")),
            "catboost_params": json.dumps(
                selection.get("composition", {}).get("catboost_params")),
            "tree_params": json.dumps(
                selection.get("process", {}).get("tree_params")),
        })
    return (pd.DataFrame(composition_rows), pd.DataFrame(process_rows),
            pd.DataFrame(audit_rows))


def with_summary(df: pd.DataFrame, group_cols, metric_cols) -> pd.DataFrame:
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["fold"] = "mean_std"
    return pd.concat([df, summary], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/cv")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    composition, process, audit = collect(Path(args.runs_root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    composition = with_summary(
        composition, ["pipeline", "head"], list(COMPOSITION_SCALAR))
    process = with_summary(
        process, ["pipeline", "head", "target"], list(PROCESS_SCALAR))

    composition.to_csv(out / "composition_heads_cv.csv", index=False)
    process.to_csv(out / "process_heads_cv.csv", index=False)
    audit.to_csv(out / "cv_selection_audit.csv", index=False)
    print(f"wrote {len(composition)} composition rows, {len(process)} process "
          f"rows, {len(audit)} audit rows -> {out}")

    headline = composition[
        (composition["fold"] == "mean_std")
        & (composition["head"] == "oof_constrained_reranker")]
    print(headline[["pipeline", "element_wape_present_mean",
                    "element_wape_all_mean",
                    "macro_false_positive_rate_mean",
                    "alloy_top1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
