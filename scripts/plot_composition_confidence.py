#!/usr/bin/env python
"""How much probability the composition head puts on the true alloy.

One raincloud per representation over the 107 held-out conditions pooled from
the five CV folds: clipped kernel density above the line, one dot per
condition below it, filled when the top-1 alloy was correct and open when it
was not. This is the continuous view of the top-1 and NLL columns of the
composition table, and unlike the per-element error it is a real distribution
rather than a few lattice-spaced values.

Writes fig_composition_confidence.png.

Usage:
  python scripts/plot_composition_confidence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icme_mg.data import ALLOYS

HEAD = "composition/oof_constrained_reranker"
FOLDS = tuple(f"cv_fold{index}" for index in range(5))
PIPELINES = ("conventional", "genai", "gnn")
BRANCH_LABELS = {"conventional": "conventional",
                 "genai": "vision embedding", "gnn": "GNN"}
BRANCH_COLOURS = {"conventional": "#3b6fb6", "genai": "#c9822f",
                  "gnn": "#4c9a56"}


def set_publish_style():
    """One type scale for the paper figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 13.5, "axes.linewidth": 1.0,
        "figure.dpi": 200,
    })


def load(pipeline: str, runs_dir: Path):
    """Probability on the true alloy and the top-1 hit, per held-out condition."""
    conditions, proba = [], []
    for fold in FOLDS:
        data = np.load(runs_dir / fold / pipeline / "predictions.npz",
                       allow_pickle=True)
        conditions.append(np.asarray(data[f"{HEAD}/condition_ids"], dtype=str))
        proba.append(data[f"{HEAD}/proba"])
    conditions = np.concatenate(conditions)
    if len(set(conditions)) != len(conditions):
        raise RuntimeError(f"{pipeline}: a condition appears in two folds")
    proba = np.concatenate(proba)
    alloys = list(ALLOYS)
    truth = np.array([alloys.index(cid.rsplit("_", 2)[0]) for cid in conditions])
    return proba[np.arange(len(truth)), truth], proba.argmax(axis=1) == truth


def plot(branches, out_path: Path, seed: int):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    set_publish_style()
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 400)
    chance = 1.0 / len(ALLOYS)

    figure, axis = plt.subplots(figsize=(12.6, 5.6))
    # Reflect at the bounds so the density does not leak outside [0, 1].
    densities = []
    for _, values, _ in branches:
        kernel = gaussian_kde(values, bw_method=0.18)
        densities.append(kernel(grid) + kernel(-grid) + kernel(2.0 - grid))
    # One scale for every row, so the heights are comparable across branches.
    tallest = max(density.max() for density in densities)
    for row, (pipeline, values, correct) in enumerate(branches):
        base = len(branches) - 1 - row
        colour = BRANCH_COLOURS.get(pipeline, "0.5")
        top = base + 0.06 + 0.55 * densities[row] / tallest
        axis.fill_between(grid, base + 0.06, top, color=colour, alpha=0.35,
                          linewidth=0)
        axis.plot(grid, top, color=colour, linewidth=2.2)
        jitter = base - 0.10 - rng.uniform(0.02, 0.24, values.size)
        axis.scatter(values[correct], jitter[correct], s=64, color=colour,
                     alpha=0.75, linewidths=0)
        axis.scatter(values[~correct], jitter[~correct], s=64,
                     facecolors="none", edgecolors=colour, linewidths=1.8,
                     alpha=0.85)
        axis.plot([np.median(values)] * 2, [base - 0.04, base + 0.10],
                  color=colour, linewidth=3.4)
        axis.text(1.005, base + 0.24,
                  f"top-1 {correct.mean():.3f}\n"
                  f"NLL {-np.log(np.clip(values, 1e-12, None)).mean():.2f}",
                  transform=axis.get_yaxis_transform(), fontsize=16,
                  va="center", color=colour)

    axis.axvline(chance, color="black", linewidth=1.6, linestyle="--")
    axis.text(chance, len(branches) - 0.30, f"chance, 1/{len(ALLOYS)}",
              fontsize=16, va="top", ha="right")
    axis.set_yticks(np.arange(len(branches))[::-1])
    axis.set_yticklabels([BRANCH_LABELS.get(pipeline, pipeline)
                          for pipeline, _, _ in branches])
    axis.set_ylim(-0.45, len(branches) - 0.15)
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("probability assigned to the true alloy")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.scatter([], [], s=44, color="0.35", linewidths=0,
                 label="top-1 correct")
    axis.scatter([], [], s=44, facecolors="none", edgecolors="0.35",
                 label="top-1 wrong")
    axis.legend(loc="upper center", frameon=True, ncol=2, fontsize=16,
                bbox_to_anchor=(0.62, 1.0))
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    parser.add_argument("--runs-dir", default="runs/cv")
    parser.add_argument("--out-dir", default="paper/ai4mat/figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branches = []
    for pipeline in args.pipelines:
        values, correct = load(pipeline, runs_dir)
        branches.append((pipeline, values, correct))
        print(f"  {pipeline}: median p(true) {np.median(values):.3f}, "
              f"top-1 {correct.mean():.3f}")

    out_path = out_dir / "fig_composition_confidence.png"
    plot(branches, out_path, args.seed)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
