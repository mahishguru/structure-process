#!/usr/bin/env python
"""Process-head response along a supervised LDA axis of the representation.

For one CV fold and one or more representation branches, this fits the two
process heads on the fold's train conditions only and plots what they predict
along a 1-D linear discriminant of the representation itself.

  fig_gnn_lda_projection.png   the 2-D velocity LDA embedding of the frozen
                               GNN condition means plus the raw velocity
                               spread along component 1.
  fig_gnn_lda_velocity.png     v_ext (left) and log v_ext (right) against the
                               velocity LDA axis of the GNN branch.
  fig_lda_temperature.png      T_ext against the temperature LDA axis, for
                               the conventional branch (left) and the vision
                               embedding branch (right).
  fig_gnn_velocity_parity.png  predicted vs true velocity, no projection.

Curves are the kernel-smoothed mean response along the axis: the heads see the
full representation plus known composition, so their raw output is not a
function of the projection alone. Train / val / test are circle / triangle /
square everywhere and only train is ever fitted on.

Usage:
  python scripts/plot_lda_process_response.py
  python scripts/plot_lda_process_response.py --pipelines gnn --cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icme_mg.data import LB, LOADERS, condition_pool, conditioned_inputs
from icme_mg.evaluation.head_metrics import process_metrics
from icme_mg.heads.process import grid_gp as grid_gp_head
from icme_mg.heads.process import trees as trees_head

PARTS = ("train", "val", "test")
MARKERS = {"train": "o", "val": "^", "test": "s"}
COLOURS = {"train": "#3b6fb6", "val": "#e8a33d", "test": "#4c9a56"}
BRANCH_LABELS = {"conventional": "conventional",
                 "genai": "vision embedding", "gnn": "GNN"}
GP_COLOUR = "#5b2d90"
GP_BAND = "#7d5ba6"
GBDT_COLOUR = "#c23b22"
Z90 = 1.6448536269514722


def set_publish_style():
    """One type scale for the paper figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 13.5, "axes.linewidth": 1.0,
        "figure.dpi": 200,
    })


def _process_score(prediction, condition_ids):
    metrics = process_metrics(prediction, condition_ids)
    return metrics["v_ext"]["mape"] + metrics["T_ext"]["wape"]


def condition_means(raw_parts):
    """One representation vector per condition, plus its true process pair."""
    output = {}
    for part, (features, condition_ids) in zip(PARTS, raw_parts):
        condition_ids = np.asarray(condition_ids).astype(str)
        conditions, pooled = condition_pool(
            np.asarray(features, dtype=float), condition_ids)
        conditions = np.asarray(conditions, dtype=str)
        output[part] = {
            "conditions": conditions, "features": pooled,
            "T_ext": LB.loc[conditions, "T_ext"].to_numpy(float),
            "v_ext": LB.loc[conditions, "v_ext"].to_numpy(float)}
    return output


def axis_candidates(features, target):
    """1-D supervised projections of the head input space.

    Alongside the discretized-target LDA variants and PLS there are two
    direct regressions on the target; their fit is itself the projection.
    Every candidate returns (axis, spread) coordinates; the spread axis is
    only used by the 2-D embedding panel.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LassoCV, RidgeCV

    edges = np.quantile(target, [1 / 3, 2 / 3])
    lda_bins = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto").fit(
            features, np.digitize(target, edges))
    lda_levels = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto").fit(
            features, np.unique(target, return_inverse=True)[1])
    pls = PLSRegression(n_components=2).fit(features, target)

    def fit_projections(model):
        direction = model.coef_.ravel()
        norm = np.linalg.norm(direction) + 1e-12
        unit = direction / norm
        # Spread axis: the PLS direction orthogonalized against the fit.
        spread = pls.x_weights_[:, 0]
        spread -= np.dot(spread, unit) * unit
        return {"axis": unit,
                "bias": float(model.intercept_) / norm,
                "spread": spread / (np.linalg.norm(spread) + 1e-12)}

    return {
        "LDA, 3 quantile bins": {"model": lda_bins},
        "LDA, one class per level": {"model": lda_levels},
        "PLS component 1": {"model": pls},
        "ridge fit": fit_projections(
            RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(features, target)),
        "lasso fit": fit_projections(
            LassoCV(n_alphas=30, max_iter=20000).fit(features, target)),
    }


def fit_axis(pooled, values):
    """Pick the 1-D projection that orders the target best on validation.

    Fitted on train conditions only; validation picks the candidate and test
    is never consulted. Component 1 is oriented so the target increases to
    the right, then standardized on train so branches share one x scale.
    """
    from scipy.stats import spearmanr

    train = pooled["train"]["features"]
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-8

    def project(model, features):
        """(axis, spread) coordinates of one candidate on one part."""
        if "model" in model:
            projected = np.atleast_2d(model["model"].transform(features))
            if projected.shape[1] < 2:
                projected = np.column_stack(
                    [projected[:, 0], np.zeros(len(projected))])
            return projected[:, :2]
        return np.column_stack(
            [features @ model["axis"] + model["bias"],
             features @ model["spread"]])

    best = None
    for name, model in axis_candidates(
            (train - mean) / scale, values["train"]).items():
        projected = {part: project(model, (pooled[part]["features"] - mean)
                                   / scale)
                     for part in PARTS}
        score = abs(spearmanr(
            projected["val"][:, 0], values["val"]).statistic)
        # Prefer simpler linear-fit axes when their validation ordering is
        # within 0.04 of the best LDA/PLS candidate; those discretized
        # candidates are fragile on small target-level counts.
        new_is_simple = name in ("ridge fit", "lasso fit")
        if best is None:
            best = (score, name, projected)
            continue
        cur_is_simple = best[1] in ("ridge fit", "lasso fit")
        if (cur_is_simple and score > best[0] + 0.04) or (
                not cur_is_simple and score > best[0] - 0.04):
            best = (score, name, projected)
    _, name, projected = best
    if np.corrcoef(projected["train"][:, 0], values["train"])[0, 1] < 0:
        for array in projected.values():
            array[:, 0] *= -1
    centre = projected["train"][:, 0].mean()
    spread = projected["train"][:, 0].std() + 1e-12
    return ({part: (array - centre) / spread
             for part, array in projected.items()}, name)


def build_axes(pipeline: str, fold: str, data):
    """Display axes for one branch, in the same space the heads take as
    input: representation + known composition + extrusion-ratio one-hot."""
    pooled = condition_means(conditioned_inputs(LOADERS[pipeline](fold))[1])
    for part in PARTS:
        if not np.array_equal(pooled[part]["conditions"],
                              np.asarray(data[f"{part}/conditions"], str)):
            raise RuntimeError(f"{pipeline}/{fold}: cached fit and axis "
                               f"disagree on the {part} conditions")
    targets = {
        "v": {part: np.log(pooled[part]["v_ext"]) for part in PARTS},
        "T": {part: pooled[part]["T_ext"] for part in PARTS},
    }
    return {key: fit_axis(pooled, values) for key, values in targets.items()}


def fit_heads(raw_parts, seed: int):
    """GBDT (val-tuned CatBoost) and the joint-grid GP, fitted on train and
    applied to all three parts."""
    process_parts = conditioned_inputs(raw_parts)[1]
    (X_train, c_train), (X_val, c_val), _ = process_parts
    fitted, params = trees_head.tune_catboost_process(
        X_train, c_train, X_val, c_val, _process_score)
    print(f"  GBDT: selected {params}", flush=True)
    gbdt = {}
    for part, (features, condition_ids) in zip(PARTS, process_parts):
        conditions, pooled = condition_pool(
            trees_head.predict_catboost_process(fitted, features),
            np.asarray(condition_ids).astype(str))
        gbdt[part] = {"conditions": np.asarray(conditions, dtype=str),
                      "T_ext": pooled[:, 0], "v_ext": pooled[:, 1]}

    print("  grid_gp: fitting joint-grid GP + constrained velocity",
          flush=True)
    run = grid_gp_head.run_grid_gp(
        raw_parts, grid_gp_head.GridGPConfig(seed=seed))
    grid = {}
    for part in PARTS:
        # The joint latent is (T_ext, log v_ext): column 0 is already in
        # physical temperature units, column 1 is in log velocity.
        mean = run["latent"][part][0]
        sigma = run["latent_sigma"][part]
        grid[part] = {
            "conditions": np.asarray(
                [str(c) for c in run["conditions"][part]], dtype=str),
            "T_ext": run["predictions"][part][:, 0],
            "v_ext": run["predictions"][part][:, 1],
            "T_mean": mean[:, 0],
            "T_low": mean[:, 0] - Z90 * sigma[:, 0],
            "T_high": mean[:, 0] + Z90 * sigma[:, 0],
            "log_mean": mean[:, 1],
            "log_low": mean[:, 1] - Z90 * sigma[:, 1],
            "log_high": mean[:, 1] + Z90 * sigma[:, 1],
        }
    return gbdt, grid, run["selected_velocity_decoder"]


def align(source, conditions):
    """Reorder a head's per-condition output onto `conditions`."""
    index = {c: i for i, c in enumerate(source["conditions"])}
    order = np.asarray([index[c] for c in conditions])
    return {key: np.asarray(values)[order]
            for key, values in source.items() if key != "conditions"}


def build(pipeline: str, fold: str, seed: int, cache_path: Path):
    print(f"[{fold}/{pipeline}]", flush=True)
    raw_parts = LOADERS[pipeline](fold)
    pooled = condition_means(raw_parts)
    gbdt, grid, decoder = fit_heads(raw_parts, seed)

    payload = {"pipeline": pipeline, "fold": fold,
               "decoder": str(decoder["decoder"]),
               "blend": float(decoder["blend"])}
    for part in PARTS:
        conditions = pooled[part]["conditions"]
        payload[f"{part}/conditions"] = conditions
        payload[f"{part}/truth_v"] = pooled[part]["v_ext"]
        payload[f"{part}/truth_T"] = pooled[part]["T_ext"]
        aligned = align(gbdt[part], conditions)
        payload[f"{part}/gbdt_v"] = aligned["v_ext"]
        payload[f"{part}/gbdt_T"] = aligned["T_ext"]
        aligned = align(grid[part], conditions)
        for key in ("T_ext", "v_ext", "T_mean", "T_low", "T_high",
                    "log_mean", "log_low", "log_high"):
            payload[f"{part}/grid_{key}"] = aligned[key]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **payload)
    print(f"  cached fit -> {cache_path}", flush=True)
    return payload


# ------------------------------------------------------------------ plotting
def smooth(x, values, grid, bandwidth, neighbours=5):
    """Gaussian-kernel local mean of `values` over the projection axis.

    The heads see the full representation, so their raw output is not a
    function of the projection alone; this is the mean response along it.
    The kernel widens to reach `neighbours` conditions where the axis is
    sparse, so the curve stays continuous without over-smoothing the dense
    stretches.
    """
    distance = np.abs(grid[:, None] - x[None, :])
    reach = np.sort(distance, axis=1)[:, min(neighbours, x.size) - 1]
    width = np.maximum(bandwidth, reach)[:, None]
    weights = np.exp(-0.5 * np.square(distance / width))
    return weights @ values / weights.sum(axis=1)


def stack(data, key, order=None):
    values = np.concatenate([data[f"{part}/{key}"] for part in PARTS])
    return values if order is None else values[order]


def projection(axis_parts):
    """Component 1 of a fitted display axis, per part."""
    return {part: values[:, 0] for part, values in axis_parts.items()}


def scatter_parts(axis, data, xs, y_key, transform=lambda v: v):
    for part in PARTS:
        axis.scatter(
            xs[part], transform(data[f"{part}/{y_key}"]),
            marker=MARKERS[part], s=88, c=COLOURS[part], edgecolors="black",
            linewidths=0.8, alpha=0.85, label=part, zorder=3)


def response(axis, data, xs, keys, bandwidth, forward=lambda v: v):
    """GP band, GP mean and GBDT mean response along the LDA axis."""
    x = np.concatenate([xs[part] for part in PARTS])
    order = np.argsort(x)
    x = x[order]
    grid = np.linspace(x.min(), x.max(), 400)
    curve = {name: smooth(x, stack(data, key, order), grid, bandwidth)
             for name, key in keys.items()}
    axis.fill_between(grid, forward(curve["low"]), forward(curve["high"]),
                      color=GP_BAND, alpha=0.16,
                      label="joint-grid GP 90% band", zorder=1)
    axis.plot(grid, forward(curve["mean"]), color=GP_COLOUR, linewidth=2.4,
              label="joint-grid GP", zorder=4)
    axis.plot(grid, forward(curve["gbdt"]), color=GBDT_COLOUR, linewidth=2.2,
              linestyle="--", label="GBDT", zorder=4)


def plot_projection(data, axis_fit, out_path: Path, fold: str):
    import matplotlib.pyplot as plt

    parts, name = axis_fit
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    embedded = np.concatenate([parts[part] for part in PARTS])
    velocity = stack(data, "truth_v")
    norm = plt.Normalize(velocity.min(), velocity.max())
    for part in PARTS:
        values = parts[part]
        axes[0].scatter(values[:, 0], values[:, 1], marker=MARKERS[part],
                        s=58, c=data[f"{part}/truth_v"], cmap="viridis",
                        norm=norm, edgecolors="black", linewidths=0.5,
                        label=part)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.2,
                    label="projection line")
    axes[0].vlines(embedded[:, 0], 0.0, embedded[:, 1], color="0.75",
                   linewidth=0.6, zorder=0)
    axes[0].set_xlabel("projection 1")
    axes[0].set_ylabel("projection 2")
    axes[0].set_title("Supervised embedding, projected onto component 1")
    axes[0].legend(loc="upper left", frameon=True)

    for part in PARTS:
        mappable = axes[1].scatter(
            parts[part][:, 0], data[f"{part}/truth_v"],
            marker=MARKERS[part], s=58, c=data[f"{part}/truth_v"],
            cmap="viridis", norm=norm, edgecolors="black", linewidths=0.5,
            label=part)
    axes[1].set_xlabel("coordinate on projection 1")
    axes[1].set_ylabel("v_ext (mm/s)")
    axes[1].set_title("Raw velocity spread along the projection line")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(alpha=0.3)
    figure.colorbar(mappable, ax=axes[1], label="v_ext (mm/s)")
    figure.suptitle(
        f"Frozen GNN velocity projection ({fold}, {name}, "
        "selected on validation)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def plot_velocity(data, axis_fit, out_path: Path, fold: str,
                  bandwidth: float, pipeline: str):
    import matplotlib.pyplot as plt

    set_publish_style()
    keys = {"mean": "grid_log_mean", "low": "grid_log_low",
            "high": "grid_log_high", "gbdt": "gbdt_v"}
    xs = projection(axis_fit[0])
    data_log = dict(data)
    for part in PARTS:
        data_log[f"{part}/gbdt_v"] = np.log(
            np.maximum(data[f"{part}/gbdt_v"], 1e-8))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, log_space in zip(axes, (False, True)):
        response(axis, data_log, xs, keys, bandwidth,
                 forward=(lambda v: v) if log_space else np.exp)
        scatter_parts(axis, data, xs, "truth_v",
                      transform=np.log if log_space else (lambda v: v))
        axis.set_xlabel(r"velocity projection 1")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(r"$v_{\mathrm{ext}}$ (mm/s)")
    axes[0].set_title("Raw velocity")
    axes[1].set_ylabel(r"$\log v_{\mathrm{ext}}$")
    axes[1].set_title("Log velocity")
    figure.legend(*axes[0].get_legend_handles_labels(),
                  loc="lower center", ncol=7, frameon=False, fontsize=16,
                  bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=[0, 0.07, 1, 1])
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def plot_temperature(panels, out_path: Path, fold: str, bandwidth: float):
    """One panel per representation branch, same axes."""
    import matplotlib.pyplot as plt

    set_publish_style()
    keys = {"mean": "grid_T_mean", "low": "grid_T_low",
            "high": "grid_T_high", "gbdt": "gbdt_T"}
    figure, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6),
                                squeeze=False)
    limits = []
    for axis, (pipeline, data, axis_fit) in zip(axes[0], panels):
        xs = projection(axis_fit[0])
        response(axis, data, xs, keys, bandwidth)
        scatter_parts(axis, data, xs, "truth_T")
        axis.set_xlabel(r"temperature projection 1")
        axis.set_title(BRANCH_LABELS.get(pipeline, pipeline))
        axis.grid(alpha=0.3)
        # Each branch has its own axis scale, so x is fitted to its own data.
        spread = np.concatenate([xs[part] for part in PARTS])
        margin = 0.04 * (spread.max() - spread.min())
        axis.set_xlim(spread.min() - margin, spread.max() + margin)
        limits.append(axis.get_ylim())
    shared_y = (min(low for low, _ in limits),
                max(high for _, high in limits))
    for axis in axes[0]:
        axis.set_ylim(shared_y)
    axes[0][0].set_ylabel(r"$T_{\mathrm{ext}}$ (${}^\circ$C)")
    figure.legend(*axes[0][0].get_legend_handles_labels(),
                  loc="lower center", ncol=7, frameon=False, fontsize=16,
                  bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=[0, 0.07, 1, 1])
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def plot_parity(data, out_path: Path, fold: str, pipeline: str):
    """Predicted against true velocity: head performance without a projection
    in the middle that can itself fail on held-out conditions."""
    import matplotlib.pyplot as plt

    set_publish_style()
    truth = stack(data, "truth_v")
    levels = np.unique(truth)
    limits = np.asarray([truth.min() / 1.35, truth.max() * 1.35])

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, log_space in zip(axes, (False, True)):
        forward = np.log if log_space else (lambda v: v)
        axis.plot(forward(limits), forward(limits), color="black",
                  linestyle="--", linewidth=1.2, label="perfect prediction",
                  zorder=1)
        for level in levels:
            axis.axvline(forward(level), color="0.85", linewidth=0.7,
                         zorder=0)
        for index, part in enumerate(PARTS):
            true_v = data[f"{part}/truth_v"]
            jitter = np.exp(0.045 * (index - 1))
            gp_x = forward(true_v * jitter * np.exp(-0.022))
            axis.vlines(gp_x, forward(np.exp(data[f"{part}/grid_log_low"])),
                        forward(np.exp(data[f"{part}/grid_log_high"])),
                        color=GP_BAND, linewidth=0.8, alpha=0.35, zorder=2)
            # The scored GP output is the catalogue-snapped value; the bar is
            # the pre-snap latent 90% interval it is decoded from.
            axis.scatter(gp_x, forward(data[f"{part}/grid_v_ext"]),
                         marker=MARKERS[part], s=88, facecolors=GP_COLOUR,
                         edgecolors="black", linewidths=0.8, alpha=0.85,
                         zorder=3, label=f"joint-grid GP, {part}")
            axis.scatter(forward(true_v * jitter * np.exp(0.022)),
                         forward(data[f"{part}/gbdt_v"]),
                         marker=MARKERS[part], s=88, facecolors=GBDT_COLOUR,
                         edgecolors="black", linewidths=0.8, alpha=0.85,
                         zorder=3, label=f"GBDT, {part}")
        axis.set_xlabel(r"true $v_{\mathrm{ext}}$ (mm/s)" if not log_space
                        else r"true $\log v_{\mathrm{ext}}$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"predicted $v_{\mathrm{ext}}$ (mm/s)")
    axes[0].set_title("Raw velocity")
    axes[1].set_ylabel(r"predicted $\log v_{\mathrm{ext}}$")
    axes[1].set_title("Log velocity")
    axes[0].legend(loc="upper left", frameon=True, fontsize=12, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def plot_temperature_parity(panels, out_path: Path, fold: str):
    """Predicted against true temperature, one panel per branch."""
    import matplotlib.pyplot as plt

    set_publish_style()
    truth = np.concatenate(
        [stack(data, "truth_T") for _, data, _ in panels])
    levels = np.unique(truth)
    span = np.asarray([truth.min() - 40, truth.max() + 40])

    figure, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6),
                                squeeze=False)
    for axis, (pipeline, data, _) in zip(axes[0], panels):
        axis.plot(span, span, color="black", linestyle="--", linewidth=1.2,
                  label="perfect prediction", zorder=1)
        for level in levels:
            axis.axvline(level, color="0.85", linewidth=0.7, zorder=0)
        for index, part in enumerate(PARTS):
            true_T = data[f"{part}/truth_T"]
            offset = 9.0 * (index - 1)
            axis.vlines(true_T + offset - 4.0, data[f"{part}/grid_T_low"],
                        data[f"{part}/grid_T_high"], color=GP_BAND,
                        linewidth=0.8, alpha=0.35, zorder=2)
            # The scored GP output is the catalogue-decoded value; the bar is
            # the latent 90% interval it is decoded from.
            axis.scatter(true_T + offset - 4.0, data[f"{part}/grid_T_ext"],
                         marker=MARKERS[part], s=88, facecolors=GP_COLOUR,
                         edgecolors="black", linewidths=0.8, alpha=0.85,
                         zorder=3, label=f"joint-grid GP, {part}")
            axis.scatter(true_T + offset + 4.0, data[f"{part}/gbdt_T"],
                         marker=MARKERS[part], s=88, facecolors=GBDT_COLOUR,
                         edgecolors="black", linewidths=0.8, alpha=0.85,
                         zorder=3, label=f"GBDT, {part}")
        axis.set_xlabel(r"true $T_{\mathrm{ext}}$ (${}^\circ$C)")
        axis.set_title(BRANCH_LABELS.get(pipeline, pipeline))
        axis.set_xlim(span)
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"predicted $T_{\mathrm{ext}}$ (${}^\circ$C)")
    axes[0][0].legend(loc="upper left", frameon=True, fontsize=12, ncol=2)
    shared = (min(axis.get_ylim()[0] for axis in axes[0]),
              max(axis.get_ylim()[1] for axis in axes[0]))
    for axis in axes[0]:
        axis.set_ylim(shared)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="cv_fold0")
    parser.add_argument("--pipelines", nargs="+",
                        default=["gnn", "conventional", "genai"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="paper/ai4mat/figures")
    parser.add_argument("--bandwidth", type=float, default=0.12,
                        help="kernel width of the projection-axis smoother")
    parser.add_argument("--cache", action="store_true",
                        help="reuse stored head fits instead of refitting")
    args = parser.parse_args()

    fits, axes_fit = {}, {}
    for pipeline in args.pipelines:
        cache_path = (Path("results/figures")
                      / f"lda_process_{pipeline}_{args.fold}.npz")
        if args.cache and cache_path.exists():
            fits[pipeline] = dict(np.load(cache_path, allow_pickle=True))
        else:
            fits[pipeline] = build(pipeline, args.fold, args.seed, cache_path)
        # The axis is cheap and independent of the head fit, so it is always
        # recomputed rather than cached.
        axes_fit[pipeline] = build_axes(pipeline, args.fold, fits[pipeline])
        for key in ("v", "T"):
            print(f"  {pipeline} {key} axis: {axes_fit[pipeline][key][1]}",
                  flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "gnn" in fits:
        plot_projection(fits["gnn"], axes_fit["gnn"]["v"],
                        out_dir / "fig_gnn_lda_projection.png", args.fold)
        written.append("fig_gnn_lda_projection.png")
    for pipeline in ("gnn", "conventional"):
        if pipeline not in fits:
            continue
        names = (f"fig_{pipeline}_lda_velocity.png",
                 f"fig_{pipeline}_velocity_parity.png")
        plot_velocity(fits[pipeline], axes_fit[pipeline]["v"],
                      out_dir / names[0], args.fold, args.bandwidth,
                      pipeline)
        plot_parity(fits[pipeline], out_dir / names[1], args.fold, pipeline)
        written += list(names)
    panels = [(pipeline, fits[pipeline], axes_fit[pipeline]["T"])
              for pipeline in ("conventional", "genai") if pipeline in fits]
    if panels:
        plot_temperature(panels, out_dir / "fig_lda_temperature.png",
                         args.fold, args.bandwidth)
        plot_temperature_parity(
            panels, out_dir / "fig_temperature_parity.png", args.fold)
        written += ["fig_lda_temperature.png", "fig_temperature_parity.png"]
    for name in written:
        print(f"wrote {out_dir}/{name}")


if __name__ == "__main__":
    main()
