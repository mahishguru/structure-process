#!/usr/bin/env python
"""Per-element composition error, pooled over the five held-out CV folds.

The reported composition head decodes to the top-1 alloy of the lattice, so
its element vector is that alloy's nominal composition. The error per element
is therefore driven by which alloy was chosen, and this figure shows the
resulting distribution per element rather than a single WAPE number.

Left: per-element WAPE where the element is present, split at zero into the
share that comes from under-predicting and the share from over-predicting, so
bar length is the reported error and the balance shows its direction.
Right: presence false-positive rate per element, the guard metric that pairs
with WAPE in the composition table.

Writes fig_composition_elements.png.

Usage:
  python scripts/plot_composition_elements.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icme_mg.data import ELEM, LB, NOMINAL

HEAD = "composition/oof_constrained_reranker"
FOLDS = tuple(f"cv_fold{index}" for index in range(5))
PIPELINES = ("conventional", "genai", "gnn")
BRANCH_LABELS = {"conventional": "conventional",
                 "genai": "vision embedding", "gnn": "GNN"}
BRANCH_COLOURS = {"conventional": "#3b6fb6", "genai": "#c9822f",
                  "gnn": "#4c9a56"}
PRESENCE_THRESHOLD = 1e-8


def load(pipeline: str, runs_dir: Path):
    """Pooled test conditions with their true and decoded element vectors."""
    conditions, predicted = [], []
    for fold in FOLDS:
        data = np.load(runs_dir / fold / pipeline / "predictions.npz",
                       allow_pickle=True)
        conditions.append(np.asarray(data[f"{HEAD}/condition_ids"], dtype=str))
        predicted.append(NOMINAL[data[f"{HEAD}/proba"].argmax(axis=1)])
    conditions = np.concatenate(conditions)
    if len(set(conditions)) != len(conditions):
        raise RuntimeError(f"{pipeline}: a condition appears in two folds")
    truth = LB.loc[conditions, list(ELEM)].to_numpy(float)
    return truth, np.concatenate(predicted)


def plot(branches, out_path: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 19, "axes.titlesize": 22, "axes.labelsize": 21,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 16, "axes.linewidth": 1.0,
        "figure.dpi": 200,
    })
    truth = branches[0][1]
    present = truth > 0
    # Elements ordered by how many held-out conditions actually contain them.
    order = np.argsort(-present.sum(axis=0))
    offsets = np.linspace(0.26, -0.26, len(branches))
    height = 0.24

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})
    for offset, (pipeline, part_truth, part_pred) in zip(offsets, branches):
        colour = BRANCH_COLOURS.get(pipeline, "0.5")
        rows_y = np.arange(len(order))[::-1] + offset
        under, over = [], []
        for index in order:
            rows = part_truth[:, index] > 0
            residual = part_pred[rows, index] - part_truth[rows, index]
            total = part_truth[rows, index].sum()
            under.append(100.0 * np.clip(-residual, 0, None).sum() / total)
            over.append(100.0 * np.clip(residual, 0, None).sum() / total)
        axes[0].barh(rows_y, -np.asarray(under), height=height, color=colour,
                     alpha=0.85, edgecolor="white", linewidth=0.5,
                     label=BRANCH_LABELS.get(pipeline, pipeline))
        axes[0].barh(rows_y, over, height=height, color=colour, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        # A branch that is exactly right for an element draws no bar at all.
        for y, total in zip(rows_y, np.asarray(under) + np.asarray(over)):
            if total < 0.05:
                axes[0].text(1.0, y, "0", color=colour, fontsize=15,
                             va="center", ha="left")
    axes[0].axvline(0.0, color="black", linewidth=1.1)
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels(
        [f"{ELEM[index]}  (n={int(present[:, index].sum())})"
         for index in order])
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{abs(value):g}"))
    axes[0].set_xlabel("WAPE (%):  under-predicted  |  over-predicted")
    axes[0].set_title("Element error (present)")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_axisbelow(True)

    for offset, (pipeline, part_truth, part_pred) in zip(offsets, branches):
        rates = []
        for index in order:
            absent = part_truth[:, index] == 0
            rates.append(
                100.0 * float((part_pred[absent, index]
                               > PRESENCE_THRESHOLD).mean())
                if absent.any() else 0.0)
        axes[1].barh(np.arange(len(order))[::-1] + offset, rates,
                     height=height, color=BRANCH_COLOURS.get(pipeline, "0.5"),
                     alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(np.arange(len(order))[::-1])
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("false positives (absent) (%)")
    axes[1].set_title("False presence")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                   frameon=True)

    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    parser.add_argument("--runs-dir", default="runs/cv")
    parser.add_argument("--out-dir", default="paper/ai4mat/figures")
    args = parser.parse_args()

    branches = []
    for pipeline in args.pipelines:
        truth, predicted = load(pipeline, Path(args.runs_dir))
        branches.append((pipeline, truth, predicted))
        # Same definition as the reported metric: WAPE per element over the
        # conditions where it is present, then averaged over elements.
        per_element = []
        for index in range(truth.shape[1]):
            rows = truth[:, index] > 0
            if rows.any():
                per_element.append(
                    np.abs(predicted[rows, index] - truth[rows, index]).sum()
                    / truth[rows, index].sum())
        print(f"  {pipeline}: {len(truth)} conditions, "
              f"present-element WAPE {100 * np.mean(per_element):.1f}%")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_composition_elements.png"
    plot(branches, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
