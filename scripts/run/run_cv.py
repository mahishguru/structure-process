#!/usr/bin/env python
"""Unified 5-fold cross-validation runner for the head benchmark.

For every (fold, representation) pair, trains and evaluates the selected head
sets on the fold's test conditions:

Task A (composition | known process):   knn, balanced_fusion,
                                        oof_constrained_reranker
Task B (process | known composition):   trees (CatBoost), gaussian_process,
                                        grid_gp (+ ordinal velocity)

Base-hyperparameter selection (CatBoost grid, kNN k/temperature) uses the
fold's validation conditions only. The reranker blend weight and the grid-GP
velocity decoder are selected on training out-of-fold predictions only. Test
conditions are touched once, for metrics.

Outputs per (fold, representation):
  runs/cv/{fold}/{pipeline}/report.json       metrics + selection audit
  runs/cv/{fold}/{pipeline}/predictions.npz   raw predictions (recompute-only)

Usage:
  python scripts/run/run_cv.py                          # everything
  python scripts/run/run_cv.py --pipelines conventional --folds loco_fold0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from icme_mg.data import (LOADERS, PIPELINES, conditioned_inputs,
                          invert_process, representation_coverage)
from icme_mg.evaluation.head_metrics import (
    composition_metrics, condition_level_process_metrics, process_metrics)
from icme_mg.heads.composition import fusion, knn as knn_head, reranker
from icme_mg.heads.process import gp as gp_head
from icme_mg.heads.process import grid_gp as grid_gp_head
from icme_mg.heads.process import trees as trees_head
from icme_mg.protocols.cv import CV_FOLDS

COMPOSITION_HEADS = ("knn", "balanced_fusion", "oof_constrained_reranker")
PROCESS_HEADS = ("trees", "gaussian_process", "grid_gp")


def _composition_score(probabilities, condition_ids):
    return composition_metrics(
        probabilities, condition_ids)["element_wape_present"]


def _process_score(prediction, condition_ids):
    metrics = process_metrics(prediction, condition_ids)
    return metrics["v_ext"]["mape"] + metrics["T_ext"]["wape"]


def run_composition(parts, raw_parts, pipeline, seed, log):
    (X_train, c_train), (X_val, c_val), (X_test, c_test) = parts

    knn_params = fusion.tune_knn(
        X_train, c_train, X_val, c_val, _composition_score)
    log(f"  knn: selected k={knn_params[0]} tau={knn_params[1]}")
    cat_params = fusion.tune_catboost_classifier(
        X_train, c_train, X_val, c_val, _composition_score)
    log(f"  catboost: selected {cat_params}")

    test_knn, _ = knn_head.knn_outputs(
        X_train, X_test, c_train, k=knn_params[0], temperature=knn_params[1])
    owner = fusion.fit_balanced_fusion(
        X_train, c_train, cat_params=cat_params, knn_params=knn_params)
    test_fusion = fusion.fusion_probabilities(owner, X_test)

    aux = "blocks" if pipeline == "conventional" else "ftt"
    rerank = reranker.run_reranker(
        raw_parts, aux, seed=seed, owner_cat_params=cat_params,
        owner_knn_params=knn_params, log=log)

    reports = {
        "knn": composition_metrics(test_knn, c_test),
        "balanced_fusion": composition_metrics(test_fusion, c_test),
        "oof_constrained_reranker": composition_metrics(
            rerank["probabilities"]["test_reranked"],
            rerank["conditions"]["test"]),
    }
    predictions = {
        "knn/proba": test_knn,
        "balanced_fusion/proba": test_fusion,
        "oof_constrained_reranker/proba": rerank["probabilities"][
            "test_reranked"],
        "oof_constrained_reranker/condition_ids": rerank["conditions"][
            "test"],
        "oof_constrained_reranker/val_proba": rerank["probabilities"][
            "val_reranked"],
        "oof_constrained_reranker/val_condition_ids": rerank["conditions"][
            "val"],
        "oof_constrained_reranker/train_oof_proba": rerank["probabilities"][
            "train_oof_reranked"],
        "oof_constrained_reranker/train_condition_ids": rerank["conditions"][
            "train"],
    }
    audit = {
        "knn_params": {"k": knn_params[0], "temperature": knn_params[1]},
        "catboost_params": cat_params,
        "reranker": {
            "auxiliary": aux,
            "selected_owner_weight": rerank["selected_owner_weight"],
            "weight_table": rerank["weight_table"],
            "block_selected_c": rerank["block_selected_c"],
            "block_search": rerank["block_search"],
            "inner_cv_protocol": rerank["inner_cv_protocol"],
        },
    }
    return reports, predictions, audit


def run_process(parts, raw_parts, seed, log):
    (X_train, c_train), (X_val, c_val), (X_test, c_test) = parts

    trees, tree_params = trees_head.tune_catboost_process(
        X_train, c_train, X_val, c_val, _process_score)
    log(f"  trees: selected {tree_params}")
    test_trees = trees_head.predict_catboost_process(trees, X_test)

    log("  gaussian_process: fitting ARD GP")
    gp = gp_head.fit_gp_process(X_train, c_train)
    gp_conditions, gp_prediction, gp_sigma, gp_uncertainty, gp_median = \
        gp_head.predict_gp_process(gp, X_test, c_test)
    gp_report = condition_level_process_metrics(gp_prediction, gp_conditions)
    for target, values in gp_uncertainty.items():
        gp_report[target].update(values)

    log("  grid_gp: fitting joint-grid GP + constrained velocity")
    grid_run = grid_gp_head.run_grid_gp(
        raw_parts, grid_gp_head.GridGPConfig(seed=seed), log=log)
    grid_report = grid_gp_head.grid_gp_report(grid_run, "test")

    reports = {
        "trees": process_metrics(test_trees, c_test),
        "gaussian_process": gp_report,
        "grid_gp": grid_report,
    }
    predictions = {
        "trees/values": test_trees,
        "gaussian_process/condition_ids": np.asarray(
            gp_conditions, dtype=object),
        "gaussian_process/mean": gp_prediction,
        "gaussian_process/median": gp_median,
        "gaussian_process/sigma": gp_sigma,
        "grid_gp/condition_ids": grid_run["conditions"]["test"],
        "grid_gp/values": grid_run["predictions"]["test"],
        "grid_gp/latent_mean": grid_run["latent"]["test"][0],
        "grid_gp/latent_covariance": grid_run["latent"]["test"][1],
    }
    audit = {
        "tree_params": tree_params,
        "grid_gp": {
            "selected_velocity_decoder": grid_run[
                "selected_velocity_decoder"],
            "velocity_search": grid_run["velocity_search"],
            "velocity_inner_cv": grid_run["velocity_inner_cv"],
        },
    }
    return reports, predictions, audit


def run_pair(pipeline: str, split: str, seed: int, out_dir: Path, log,
             tasks=("composition", "process")):
    raw_parts = LOADERS[pipeline](split)
    coverage = representation_coverage(split, raw_parts)
    missing = [c for part in ("val", "test")
               for c in coverage[part]["missing_conditions"]]
    if missing:
        raise RuntimeError(
            f"{pipeline}/{split}: incomplete val/test coverage: {missing}")
    train_missing = coverage["train"]["missing_conditions"]
    if train_missing:
        log(f"warning: {len(train_missing)} train conditions have no "
            f"{pipeline} representation (frozen cache): {train_missing}")

    composition_parts, process_parts = conditioned_inputs(raw_parts)
    composition, composition_pred, composition_audit = {}, {}, {}
    process, process_pred, process_audit = {}, {}, {}
    if "composition" in tasks:
        log(f"  task A: composition heads "
            f"({len(composition_parts[0][1])} train "
            f"images, {len(composition_parts[2][1])} test images)")
        composition, composition_pred, composition_audit = run_composition(
            composition_parts, raw_parts, pipeline, seed, log)
    if "process" in tasks:
        log(f"  task B: process heads")
        process, process_pred, process_audit = run_process(
            process_parts, raw_parts, seed, log)

    report = {
        "protocol": "condition_grouped_5fold_cv",
        "split": split,
        "pipeline": pipeline,
        "seed": seed,
        "input_contract": {
            "composition": ("representation + known_T_ext_and_log_v_ext + "
                            "extrusion_ratio_type_one_hot"),
            "process": ("representation + known_element_wt_percent + "
                        "extrusion_ratio_type_one_hot"),
        },
        "representation_coverage": coverage,
        "composition_heads": composition,
        "process_heads": process,
        "selection": {"composition": composition_audit,
                      "process": process_audit},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    prediction_path = out_dir / "predictions.npz"
    if set(tasks) != {"composition", "process"} and report_path.exists():
        # Partial re-run: keep the untouched task exactly as it was written.
        previous = json.loads(report_path.read_text())
        stored = dict(np.load(prediction_path, allow_pickle=True))
        for task, key in (("composition", "composition_heads"),
                          ("process", "process_heads")):
            if task in tasks:
                continue
            report[key] = previous[key]
            report["selection"][task] = previous["selection"][task]
        keep = {k: v for k, v in stored.items()
                if not any(k.startswith(f"{task}/") for task in tasks)}
    else:
        keep = {"image_condition_ids": np.asarray(
            composition_parts[2][1], dtype=object)}
    report_path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        prediction_path, **keep,
        **{f"composition/{k}": v for k, v in composition_pred.items()},
        **{f"process/{k}": v for k, v in process_pred.items()})
    log(f"  wrote {out_dir}/report.json + predictions.npz")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=list(CV_FOLDS))
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    parser.add_argument("--out-root", default="runs/cv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=["composition",
                                                       "process"],
                        choices=["composition", "process"])
    args = parser.parse_args()

    for split in args.folds:
        for pipeline in args.pipelines:
            out_dir = Path(args.out_root) / split / pipeline
            if args.skip_existing and (out_dir / "report.json").exists():
                print(f"skip {split}/{pipeline} (report exists)", flush=True)
                continue
            banner = f"=== {split} / {pipeline} ==="
            print(f"\n{banner}", flush=True)
            started = time.time()
            log = lambda message: print(f"[{split}/{pipeline}] {message}",
                                        flush=True)
            report = run_pair(pipeline, split, args.seed, out_dir, log,
                              tuple(args.tasks))
            elapsed = (time.time() - started) / 60
            rerank = report["composition_heads"]["oof_constrained_reranker"]
            grid = report["process_heads"]["grid_gp"]
            print(
                f"{split}/{pipeline}: reranker present-WAPE "
                f"{rerank['element_wape_present']:.3f} (all "
                f"{rerank['element_wape_all']:.3f}, FP "
                f"{rerank['macro_false_positive_rate']:.3f}) | "
                f"grid_gp T MAPE {grid['T_ext']['mape']:.3f} v MAPE "
                f"{grid['v_ext']['mape']:.3f} (log MAE "
                f"{grid['log_v_ext']['mae']:.3f}) | {elapsed:.1f} min",
                flush=True)


if __name__ == "__main__":
    main()
