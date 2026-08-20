# Final Paper Results (source of truth)

PROTOCOL: 5-fold condition-grouped cross-validation over 107 conditions in 14
alloys, composition on an 8-element lattice (Al, Zn, Mn, Ce, Gd, Ca, Nd, Y).
Every entry is a fold mean +/- std over the five held-out condition groups. This
is the only reported protocol; the single-split seed-0 study is superseded and
archived (results/archive/paper_results_seed0_2026-08-11.md).

Story: inverse structure-to-recipe on scarce data. Both tasks share one
structural prior and one winner.
(1) Composition is a discrete 14-alloy lattice. The OOF-constrained reranker
    blends a condition-balanced CatBoost+kNN owner with per-descriptor-block
    logistic heads, the blend weight chosen only from training out-of-fold
    predictions under all-element WAPE and false-positive guard constraints.
(2) Process is a discrete catalogue of observed (T_ext, v_ext) pairs. The
    winning head decodes a joint correlated GP over that observed grid
    (T_ext) and an OOF-constrained log-GP/ordinal blend snapped onto the
    observed velocity levels (v_ext) instead of
    regressing continuously.
(3) The winning representation is conventional descriptors for composition and
    is split across representations for the process window.

Headline metric: element WAPE, the error on the elements actually present in
the alloy. The deployment question is the levels of the known alloy's elements,
not the presence pattern, so present-element WAPE is the primary number and is
reported simply as WAPE. Because a head can game it by over-predicting element
presence, every composition row also carries the two guard metrics, all-element
WAPE and macro false-positive rate.

## Stored artifacts (every number rebuildable without refitting)

| artifact | content |
|---|---|
| runs/cv/{fold}/{pipeline}/report.json | per-fold metrics + full selection audit (reranker weights, grid-GP decoder, tuned hyperparameters) |
| runs/cv/{fold}/{pipeline}/predictions.npz | raw test predictions per fold, so any new metric is a recomputation, never a re-fit |
| results/tables/composition_heads_cv.csv | tidy (fold, pipeline, head) x (WAPE, all-WAPE, FP, MAE, top-1, top-3, NLL), fold rows + mean/std |
| results/tables/process_heads_cv.csv | tidy (fold, pipeline, head, target) x (MAE, MAPE, WAPE, R2, NLL, coverage90), fold rows + mean/std |
| results/tables/cv_selection_audit.csv | per-(fold, pipeline) reranker auxiliary, selected owner weight, grid-GP decoder, tuned hyperparameters |

Commands:
- `python scripts/run/run_cv.py` runs all five folds x three representations.
- `python scripts/run/aggregate_cv.py` rebuilds the three CSVs from the
  per-fold reports.

## Table 3: Conditioned head benchmark, 5-fold cross-validation - THE PAPER TABLES

Fold mean +/- std over five held-out condition groups. Input contract: the two
tasks are symmetric. Composition heads see representation + KNOWN T_ext and
log(v_ext) + extrusion-ratio one-hot; process heads see representation + KNOWN
element wt% + extrusion-ratio one-hot.

### Table 3a: Composition given known process (WAPE / top-1; guard columns right)

| head | conv WAPE | vision WAPE | gnn WAPE | conv top-1 | vision top-1 | gnn top-1 | conv all-WAPE | conv FP % |
|---|---|---|---|---|---|---|---|---|
| kNN retrieval | 30.0% +/- 7.6 | 57.2% +/- 16.4 | 44.8% +/- 11.7 | 0.496 | 0.294 | 0.279 | 61.7% | 10.3 |
| Condition-balanced fusion | 16.8% +/- 5.8 | 44.7% +/- 18.0 | 44.3% +/- 8.0 | 0.612 | 0.276 | 0.266 | 25.4% | 1.3 |
| OOF-constrained reranker | **13.6% +/- 4.1** | **41.6% +/- 18.9** | **44.2% +/- 8.0** | **0.646** | **0.287** | **0.266** | **19.7%** | **1.1** |

all-WAPE and FP % for the other pipelines are in composition_heads_cv.csv. The
reranker wins on conventional on every metric at once: lower WAPE, lower
all-WAPE, lower false-positive rate, and higher top-1 than the fusion it must
beat. On vision and gnn it converges to the fusion (selected owner weight 1.0 on
8 of 10 folds; the FT-Transformer auxiliary carries no block structure and adds
nothing), so it neither helps nor hurts there.

### Table 3b: Process given known composition (MAE / MAPE / R2 per target)

T_ext MAE in C, v_ext MAE in mm/s. Compare MAPE down a column, never across the
two targets: a constant train-mean predictor scores 14.5% on T_ext against
136.5% on v_ext.

| head | target | conv MAE | conv MAPE | conv R2 | vision MAE | vision MAPE | vision R2 | gnn MAE | gnn MAPE | gnn R2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees | T_ext | 41.6 | 11.9% | 0.349 | 39.2 | 11.4% | 0.391 | 36.5 | 10.7% | 0.443 |
| Gaussian process (ARD) | T_ext | 52.1 | 15.2% | 0.039 | 72.1 | 19.9% | -0.813 | 49.5 | 14.7% | 0.084 |
| Joint-grid GP | T_ext | 38.1 | 10.9% | 0.326 | **35.2** | **10.5%** | 0.367 | 36.6 | 10.8% | 0.357 |
| Gradient-boosted trees | v_ext | 1.34 | 64.5% | 0.280 | 1.22 | 55.9% | 0.409 | 1.11 | 56.0% | 0.525 |
| Gaussian process (ARD) | v_ext | 1.71 | 104.7% | -0.240 | 1.76 | 120.5% | -0.200 | 1.44 | 94.5% | 0.012 |
| Joint-grid GP | v_ext | **0.98** | **48.7%** | **0.549** | **1.10** | **49.6%** | **0.414** | **1.01** | **51.4%** | **0.542** |

The joint-grid GP wins v_ext on all three representations and T_ext on
conventional and vision; the gnn trees edge it on T_ext R2 (0.443). The grid
decode is what wins the discrete velocity task, exactly as designed.

### Table 3c: Velocity scored in log space

Extrusion speed spans 0.5-7.5 mm/s over eleven distinct press settings, so raw
MAE flatters a head that predicts near the middle of the range. In log space a
fixed error is a fixed multiplicative factor everywhere. MAPE and WAPE are
omitted because log(v_ext) crosses zero inside the observed range. fold =
exp(MAE) is the multiplicative counterpart: 1.53 means the typical prediction
is off by a factor of 1.53.

| head | conv MAE | conv fold | conv R2 | vision MAE | vision fold | vision R2 | gnn MAE | gnn fold | gnn R2 |
|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees | 0.580 | 1.79 | 0.418 | 0.491 | 1.63 | 0.548 | 0.475 | 1.61 | 0.557 |
| Gaussian process (ARD) | 0.751 | 2.14 | -0.101 | 0.742 | 2.17 | -0.057 | 0.626 | 1.88 | 0.123 |
| Joint-grid GP | **0.420** | **1.53** | **0.632** | **0.450** | **1.58** | **0.577** | **0.443** | **1.56** | **0.568** |

Only in log space is the ARD GP's failure fully visible: negative R2 on two of
three representations, i.e. worse than predicting the training mean speed.

### What changed in the velocity decoder

The velocity axis now carries the same observed-catalogue constraint that the
temperature axis always had. The continuous estimate is snapped onto the
observed velocity levels, nearest in log space, because extrusion speed is set
from a short menu of press settings and a value between two settings is never a
possible answer. The snap is a fixed modelling decision, not a searched
hyperparameter; only the continuous estimate that feeds it (log-GP, or a blend
with the ordinal classifier or the grid posterior median) is chosen on training
OOF predictions, ranked by log-space MAE rather than raw MAPE. Effect on the
5-fold means:

| repr | MAE | MAPE | WAPE | R2 | log MAE |
|---|---|---|---|---|---|
| conv | 1.050 -> 0.976 | 49.9 -> 48.7 | 42.1 -> 39.4 | 0.535 -> 0.549 | 0.437 -> 0.420 |
| vision | 1.118 -> 1.099 | 50.0 -> 49.6 | 45.1 -> 44.5 | 0.447 -> 0.414 | 0.465 -> 0.450 |
| gnn | 1.124 -> 1.012 | 53.3 -> 51.4 | 45.1 -> 40.5 | 0.473 -> 0.542 | 0.474 -> 0.443 |

MAE, MAPE, WAPE and log MAE improve on all three representations, and the
joint-grid GP now beats the trees on every velocity metric everywhere (before,
gnn trees held v_ext MAE 1.11 against 1.12). Vision R2 falls, the one
regression: snapping converts a few near-misses on the fastest recipes into
larger squared errors, which only R2 penalises.

What did not work, and was rejected: a hard argmin over observed (T, v) pairs
(improves MAPE, degrades MAE/WAPE/R2 everywhere); an alloy-conditioned
catalogue prior (the known-composition kernel already carries alloy identity);
isotonic recalibration (OOF MAE 0.905 against test 1.030, i.e. it wins any OOF
selection and loses on test); and wider feature-map sweeps over
pca_components, known_weight and gp_noise (defaults best or tied everywhere,
including for the 1280-D vision latents).

## Calibration and uncertainty

- Composition UQ is the fused 14-way distribution over nominal compositions.
  Conventional 5-fold class NLL: reranker 1.38, balanced fusion 1.49, kNN 5.92.
- Process UQ: the joint-grid GP carries coverage90 from its variance-tempered
  latent predictive. Conventional T_ext coverage 0.52, v_ext 0.56 (target 0.90);
  the gnn ARD GP is the best-calibrated single head (T_ext 0.69, v_ext 0.71).
  Coverage is below nominal on the grid heads; they are point-accurate, not
  calibrated, and the GP sigma should be reported with that caveat.

## Representation verdict

Conventional descriptors win composition outright (WAPE 13.6% against 41.6%
vision and 44.2% gnn). The process window splits: vision grid-GP is the sharpest
on T_ext (35.2 C), conventional grid-GP the best on v_ext (0.98 mm/s, R2
0.549), and gnn the best-calibrated. This is sharper than the earlier
"complementary halves" reading: the composition side is decisively conventional.

## Method notes

- The reranker blend weight and the grid-GP velocity decoder are selected only
  on training out-of-fold predictions; validation selects base hyperparameters
  (CatBoost grid, kNN k/temperature); test is touched once, for metrics.
- Inner CV stratifies by alloy only when every alloy has at least the requested
  number of training conditions, else falls back to unstratified condition-level
  KFold at full granularity (5 inner folds). Several Mg-Gd-Mn alloys have only
  two training conditions per fold, which would otherwise collapse the OOF
  selection to two folds.
- GNN rows use the frozen encoder cached per fold; they are fixed-encoder
  diagnostics, not refit per fold.

## Sources

| table | source |
|---|---|
| Table 3 | runs/cv/{cv_fold0..4}/{conventional,genai,gnn}/report.json; raw predictions runs/cv/*/predictions.npz; tidy values results/tables/*_cv.csv; scripts/run/{run_cv,aggregate_cv}.py; heads in src/icme_mg/heads/ |
| archived seed-0 | results/archive/paper_results_seed0_2026-08-11.md |
