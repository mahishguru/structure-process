"""Recompute every metric from stored predictions, refitting nothing.

The benchmark writes raw per-image test predictions next to each summary, so a
new metric never requires re-training the heads.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_pipeline_heads as heads  # noqa: E402


def recompute(path: Path) -> dict:
    store = np.load(path, allow_pickle=True)
    condition_ids = store["image_condition_ids"].astype(str)
    composition = {}
    process = {}
    for key in store.files:
        parts = key.split("/")
        if len(parts) != 3:
            continue
        group, head, kind = parts
        if group == "composition" and kind == "proba":
            composition[head] = heads.composition_metrics(
                store[key], condition_ids)
        elif group == "composition" and kind == "elements":
            composition[head] = heads.regression_composition_metrics(
                store[key], condition_ids)
        elif group == "process" and kind == "values":
            process[head] = heads.process_metrics(store[key], condition_ids)

    gp_conditions = store["process/gaussian_process/condition_ids"].astype(str)
    gp_mean = store["process/gaussian_process/mean"]
    gp_sigma = store["process/gaussian_process/sigma"]
    # Point metrics use the smeared lognormal mean; the density is evaluated
    # at the median, which is the GP's predictive mean in transformed space.
    gp_center = (store["process/gaussian_process/median"]
                 if "process/gaussian_process/median" in store.files
                 else gp_mean)
    gp_report = heads.condition_level_process_metrics(gp_mean, gp_conditions)
    # The GP is Gaussian in the transformed space (log for v_ext), so the
    # z-score and the density must be evaluated there.
    truth = heads.process_targets(list(gp_conditions))
    transformed_mean = heads.transform_process(gp_center)
    for index, target in enumerate(heads.PROCESS_COLUMNS):
        z_score = (truth[:, index] - transformed_mean[:, index]) \
            / gp_sigma[:, index]
        jacobian = truth[:, index].mean() if target == "v_ext" else 0.0
        gp_report[target]["nll"] = float(np.mean(
            0.5 * np.square(z_score) + np.log(gp_sigma[:, index])
            + 0.5 * np.log(2 * np.pi)) + jacobian)
        gp_report[target]["coverage90"] = float(
            np.mean(np.abs(z_score) < 1.6449))
    process["gaussian_process"] = gp_report
    return {"composition_heads": composition, "process_heads": process}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random_seed0")
    parser.add_argument(
        "--pipelines", nargs="+", default=["conventional", "genai", "gnn"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {
        pipeline: recompute(Path(
            f"runs/predictions_cross_pipeline_{pipeline}_{args.split}.npz"))
        for pipeline in args.pipelines
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
