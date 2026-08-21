#!/usr/bin/env python
"""Where alloy identification goes wrong, per alloy and per representation.

Pools the five condition-grouped CV test folds, so every one of the 107
conditions is scored exactly once by a model that never saw it, and splits the
top-1 alloy prediction into four chemistry-aware outcomes:

  exact              the held-out alloy itself
  one lattice step   an immediate neighbour in the alloy lattice: one Gd step
                     or one Mn step inside the Mg-Gd(-Mn) grid, or another
                     member of the dilute-Zn trio
  same family        the right family, two or more steps away
  other family       a different alloy family altogether

The point of the split is that a bare top-1 hides whether the misses are
near misses. Writes fig_composition_taxonomy.png.

Usage:
  python scripts/plot_composition_taxonomy.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icme_mg.data import ALLOYS, LB

HEAD = "composition/oof_constrained_reranker"
FOLDS = tuple(f"cv_fold{index}" for index in range(5))
PIPELINES = ("conventional", "genai", "gnn")
BRANCH_LABELS = {"conventional": "conventional",
                 "genai": "vision embedding", "gnn": "GNN"}
OUTCOMES = ("exact", "one lattice step", "same family", "other family")
OUTCOME_COLOURS = {"exact": "#2f6f4e", "one lattice step": "#8fbf7f",
                   "same family": "#e8a33d", "other family": "#b5423a"}

# Mg-Gd(-Mn) grid coordinates; the dilute-Zn trio and the two singletons are
# handled by family membership alone.
GD_GRID = {
    "Mg-2Gd": (0, 0), "Mg-2Gd-0.5Mn": (0, 1), "Mg-2Gd-1Mn": (0, 2),
    "Mg-5Gd": (1, 0), "Mg-5Gd-0.5Mn": (1, 1), "Mg-5Gd-1Mn": (1, 2),
    "Mg-10Gd": (2, 0), "Mg-10Gd-0.5Mn": (2, 1), "Mg-10Gd-1Mn": (2, 2),
}
ZN_TRIO = ("Z1", "ZNd10", "ZX10")
FAMILIES = {**{name: "Mg-Gd(-Mn)" for name in GD_GRID},
            **{name: "dilute Zn" for name in ZN_TRIO},
            "AZ31": "AZ31", "ME21": "ME21"}
# Display order: the Gd grid by Gd level then Mn level, then the Zn trio.
ORDER = (list(GD_GRID) + list(ZN_TRIO) + ["AZ31", "ME21"])


def base_name(alloy: str) -> str:
    return alloy.replace("_extruded", "")


def classify(true_alloy: str, predicted: str) -> str:
    if true_alloy == predicted:
        return "exact"
    if FAMILIES[true_alloy] != FAMILIES[predicted]:
        return "other family"
    if true_alloy in GD_GRID:
        step = sum(abs(a - b) for a, b in zip(GD_GRID[true_alloy],
                                              GD_GRID[predicted]))
        return "one lattice step" if step == 1 else "same family"
    # The three dilute-Zn alloys differ by a single minor addition.
    return "one lattice step"


def load_predictions(pipeline: str, runs_dir: Path):
    """Pooled test predictions over the five folds, one row per condition."""
    conditions, predicted = [], []
    for fold in FOLDS:
        path = runs_dir / fold / pipeline / "predictions.npz"
        data = np.load(path, allow_pickle=True)
        conditions.append(np.asarray(data[f"{HEAD}/condition_ids"], dtype=str))
        predicted.append(data[f"{HEAD}/proba"].argmax(axis=1))
    conditions = np.concatenate(conditions)
    if len(set(conditions)) != len(conditions):
        raise RuntimeError(f"{pipeline}: a condition appears in two folds")
    lattice = np.asarray([base_name(name) for name in ALLOYS])
    return conditions, lattice[np.concatenate(predicted)]


def tabulate(conditions, predicted):
    """Outcome counts per true alloy, plus the per-alloy condition count."""
    truth = np.asarray([base_name(LB.loc[cid].name.rsplit("_", 2)[0])
                        for cid in conditions])
    counts = {alloy: dict.fromkeys(OUTCOMES, 0) for alloy in ORDER}
    for true_alloy, prediction in zip(truth, predicted):
        counts[true_alloy][classify(true_alloy, prediction)] += 1
    total = {alloy: sum(counts[alloy].values()) for alloy in ORDER}
    return counts, total


def plot(tables, out_path: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 13.5, "axes.linewidth": 1.0,
        "figure.dpi": 200,
    })
    height = 0.72
    positions = np.arange(len(ORDER))[::-1]
    figure, axes = plt.subplots(1, len(tables), figsize=(5.6 * len(tables), 6),
                                sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for axis, (pipeline, counts, total) in zip(axes, tables):
        left = np.zeros(len(ORDER))
        for outcome in OUTCOMES:
            widths = np.asarray(
                [counts[alloy][outcome] / max(total[alloy], 1)
                 for alloy in ORDER])
            axis.barh(positions, widths, height=height, left=left,
                      color=OUTCOME_COLOURS[outcome], edgecolor="white",
                      linewidth=0.6, label=outcome)
            left += widths
        exact = sum(counts[alloy]["exact"] for alloy in ORDER)
        near = sum(counts[alloy]["one lattice step"] for alloy in ORDER)
        pool = sum(total.values())
        axis.set_title(f"{BRANCH_LABELS.get(pipeline, pipeline)}\n"
                       f"top-1 {exact / pool:.2f}, "
                       f"within one step {(exact + near) / pool:.2f}\n",
                       fontsize=15)
        axis.set_xlim(0, 1)
        axis.set_xlabel("fraction of held-out conditions")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        [f"{alloy}  (n={tables[0][2][alloy]})" for alloy in ORDER])
    for boundary in (len(GD_GRID) - 0.5, len(GD_GRID) + len(ZN_TRIO) - 0.5):
        for axis in axes:
            axis.axhline(len(ORDER) - 1 - boundary, color="0.4",
                         linewidth=0.9, linestyle=":")
    figure.legend(*axes[0].get_legend_handles_labels(),
                  loc="outside lower center", ncol=len(OUTCOMES),
                  frameon=False, fontsize=18)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    parser.add_argument("--runs-dir", default="runs/cv")
    parser.add_argument("--out-dir", default="paper/ai4mat/figures")
    args = parser.parse_args()

    tables = []
    for pipeline in args.pipelines:
        conditions, predicted = load_predictions(pipeline,
                                                 Path(args.runs_dir))
        counts, total = tabulate(conditions, predicted)
        tables.append((pipeline, counts, total))
        pool = sum(total.values())
        exact = sum(counts[alloy]["exact"] for alloy in ORDER)
        near = sum(counts[alloy]["one lattice step"] for alloy in ORDER)
        print(f"  {pipeline}: {pool} conditions, top-1 {exact / pool:.3f}, "
              f"within one step {(exact + near) / pool:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_composition_taxonomy.png"
    plot(tables, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
