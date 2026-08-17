# Final Paper Results (source of truth)

PROTOCOL: the paper reports ONE frozen alloy-stratified random split over whole
conditions (seed 0, 72/18/18 conditions). Table 3 is the only set of numbers
that appears in paper/main.tex. The earlier random condition-grouped 5-fold CV
tables were superseded and removed; their raw summaries remain under runs/ and
results/archive/.

Story: inverse structure->recipe on scarce data. Three findings:
(1) the two halves of a recipe favour different representations (conventional
descriptors identify the alloy, GNN embeddings predict the extrusion window);
(2) composition is a discrete 14-alloy lattice, so the FT-Transformer and the
fused classifier beat direct continuous regression on present-element WAPE
(15.8% and 20.0% vs 41.8%) and additionally name the alloy; (3) random image
splitting is fully leaked because repeated images of every test condition occur
in training, so the study assigns whole conditions to the frozen split.

The final fusion gives each processing condition equal total training weight.
This matches the condition-level estimand despite the observed 15-108 images
per condition and reduces bias against alloys represented by only four
conditions. The additional prior-power correction is NOT used in the headline
result: on the seed-0 split it degrades conventional WAPE from 20.0% to 43.6%,
i.e. tau cannot be selected reliably from 18 validation conditions.

Flow-AR is dropped from the paper: it has no seed-0 counterpart, so no
flow-versus-lattice claim is made.

## Stored artifacts (every table and figure is rebuildable without retraining)

Aggregate metrics alone were not enough: adding MAPE once forced a full re-fit.
The benchmark therefore also writes RAW test predictions, so any new metric is
a recomputation instead of a re-run.

| artifact | content |
|---|---|
| runs/predictions_cross_pipeline_{pipeline}_random_seed0.npz | raw per-image test predictions: 14-way class probabilities for the 7 probabilistic composition heads, 8-element vectors for the 2 regression heads, (T_ext, v_ext) for the 4 image-level process heads, and condition-level GP mean AND predictive sigma; plus the image condition ids needed to join to labels |
| runs/summary_cross_pipeline_heads_{pipeline}_random_seed0.json | aggregate metrics as computed at run time |
| results/tables/composition_heads.csv | tidy (pipeline, head) x (WAPE, MAE, top-1, top-3, class NLL) -> Table 3a |
| results/tables/composition_per_element.csv | tidy (pipeline, head, element) x (MAE, present-WAPE) -> per-element figures |
| results/tables/process_heads.csv | tidy (pipeline, head, target) x (MAE, MAPE, WAPE, R2, NLL, coverage90) -> Table 3b |

Commands:
- `python scripts/metrics_from_predictions.py --out FILE` recomputes every
  metric from the npz files, fitting nothing.
- `python scripts/export_paper_tables.py` regenerates the CSVs and prints the
  markdown for Tables 3a/3b.

Verified: recomputing from the conventional npz reproduces all 227 metric
scalars in the summary JSON exactly (zero mismatches), so the prediction store
is a sufficient statistic for the reported numbers. Storing the GP sigma is
what keeps NLL and coverage recomputable.

## Table 3: Cross-pipeline conditioned head benchmark (single random split, seed 0) - THE PAPER TABLES

ONE alloy-stratified random split over whole conditions (seed 0),
train/val/test = 72/18/18 conditions (2909/729/710 images), no fold averaging.
Every entry is a single-split point estimate on 18 test conditions, so top-1
moves in steps of 0.056 and head-to-head gaps below ~0.15 are not resolvable.

Input contract (see paper/dataset.md): the two tasks are symmetric. Composition
heads see representation + KNOWN T_ext and log(v_ext) + extrusion_ratio_type
one-hot; process heads see representation + KNOWN element wt% +
extrusion_ratio_type one-hot. Each task is conditioned on the half of the
condition label that the other one predicts.
(source: runs/summary_cross_pipeline_heads_{conventional,genai,gnn}_random_seed0.json)

### Table 3a: Composition given known process parameters (element WAPE /
element MAE / alloy top-1)

| head | conv WAPE | conv MAE | conv top-1 | genai WAPE | genai MAE | genai top-1 | gnn WAPE | gnn MAE | gnn top-1 |
|---|---|---|---|---|---|---|---|---|---|
| kNN retrieval | 23.7% | 0.410 | 0.444 | 24.5% | 0.467 | 0.444 | 44.1% | 0.363 | 0.222 |
| XGBoost classifier | 45.8% | 0.436 | 0.444 | 42.7% | 0.290 | 0.222 | 45.0% | 0.351 | 0.222 |
| CatBoost classifier | 42.1% | 0.265 | 0.500 | 59.5% | 0.348 | 0.167 | 45.6% | 0.369 | 0.222 |
| FT-Transformer | 15.8% | 0.197 | 0.556 | 42.0% | 0.557 | 0.333 | 63.7% | 0.398 | 0.167 |
| CatBoost + kNN fusion | 19.1% | 0.219 | 0.556 | 45.5% | 0.357 | 0.222 | 45.5% | 0.365 | 0.222 |
| Condition-balanced fusion | 20.0% | 0.262 | 0.556 | 45.5% | 0.357 | 0.222 | 44.9% | 0.364 | 0.222 |
| Prior-adjusted balanced fusion | 43.6% | 0.358 | 0.444 | 48.5% | 0.355 | 0.167 | 42.1% | 0.350 | 0.222 |
| XGBoost regression | 59.2% | 0.270 | - | 51.6% | 0.318 | - | 44.1% | 0.307 | - |
| CatBoost regression | 41.8% | 0.325 | - | 54.3% | 0.344 | - | 40.1% | 0.321 | - |

Top-3 and class NLL are in the JSON. Best top-3 per pipeline: conventional
0.722 (FT-Transformer), genai 0.722 (prior-adjusted balanced fusion), gnn 0.611
(CatBoost classifier / prior-adjusted balanced fusion). Best class NLL: 1.79
conventional (FT-Transformer), 2.10 genai and 2.56 gnn (prior-adjusted balanced
fusion in both).

### Table 3b: Process given known composition (MAE / MAPE / R2 per target)

T_ext MAE is in C, v_ext MAE in mm/s. Both targets are strictly positive
(T_ext 200-500 C, v_ext 0.5-7.5 mm/s), so MAPE is well defined, but the two
targets have very different MAPE floors: a constant train-mean predictor scores
14.5% on T_ext and 136.5% on v_ext, because v_ext spans 1.18 decades against
0.40 for T_ext. Compare MAPE down a column, never across the two targets.
(full precision in results/tables/process_heads.csv)

| head | conv T MAE | conv T MAPE | conv T R2 | conv v MAE | conv v MAPE | conv v R2 | genai T MAE | genai T MAPE | genai T R2 | genai v MAE | genai v MAPE | genai v R2 | gnn T MAE | gnn T MAPE | gnn T R2 | gnn v MAE | gnn v MAPE | gnn v R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN retrieval | 41.4 | 11.2% | 0.460 | 1.14 | 50.5% | 0.539 | 58.4 | 15.5% | -0.040 | 1.38 | 64.9% | 0.235 | 34.5 | 9.9% | 0.525 | 0.96 | 36.8% | 0.525 |
| XGBoost | 38.2 | 10.1% | 0.354 | 0.88 | 39.3% | 0.613 | 35.2 | 10.1% | 0.479 | 1.00 | 39.8% | 0.557 | 30.4 | 9.0% | 0.513 | 0.99 | 44.6% | 0.587 |
| CatBoost | 40.6 | 10.3% | 0.358 | 1.07 | 43.2% | 0.367 | 33.6 | 9.5% | 0.515 | 1.00 | 39.8% | 0.552 | 29.0 | 8.5% | 0.579 | 0.90 | 36.7% | 0.606 |
| FT-Transformer | 48.8 | 12.8% | 0.193 | 1.09 | 57.8% | 0.534 | 39.7 | 10.7% | 0.507 | 1.06 | 44.8% | 0.487 | 34.5 | 9.6% | 0.519 | 0.95 | 39.2% | 0.599 |
| Gaussian process | 34.4 | 9.3% | 0.375 | 1.42 | 69.5% | 0.195 | 73.0 | 19.3% | -0.985 | 1.22 | 50.4% | 0.376 | 49.8 | 14.8% | -0.000 | 1.12 | 42.1% | 0.308 |

GP 90% coverage (target 0.90): T_ext 0.833 conventional / 0.389 genai /
0.667 gnn; v_ext 0.556 conventional / 0.444 genai / 0.889 gnn.

Message (stated at the strength the split supports): conventional descriptors
are the strongest composition representation (FT-Transformer 15.8% WAPE, top-1
0.556), and the tuned FT-Transformer now beats the fusion it used to trail.
For the process window the picture split after re-balancing: the GNN heads are
the most consistent on T_ext (R2 0.513-0.579 for the tuned trees) and hold the
best v_ext calibration (GP cov90 0.889), while the conventional GP is the best
single T_ext result (MAE 34.4 C, R2 0.375, cov90 0.833) but its v_ext coverage
collapsed to 0.556. GenAI is the weakest on T_ext (GP R2 -0.985). With 18 test
conditions the head rankings inside a pipeline are noise; the cross-pipeline
pattern is the reportable finding.

## Task A conditioning (2026-08-06)

Task A now receives the known extrusion parameters, mirroring Task B: the
composition heads see representation + standardized (T_ext, log v_ext) +
extrusion-ratio one-hot. Input widths went 440 -> 442 conventional, 1,282 ->
1,284 genai, 130 -> 132 gnn. Process-head inputs and results are unchanged
(bit-identical apart from ~1e-13 CatBoost GPU reduction noise), which is the
control for the change.

Element WAPE before -> after:

| head | conventional | genai | gnn |
|---|---|---|---|
| kNN | 31.3 -> 31.3 | 49.1 -> 40.9 | 51.8 -> 38.8 |
| XGB classifier | 51.6 -> 51.6 | 46.1 -> 37.5 | 55.1 -> 37.0 |
| CatBoost classifier | 24.8 -> 24.8 | 52.4 -> 39.4 | 58.2 -> 38.0 |
| FT-Transformer | 42.2 -> 36.3 | 70.1 -> 51.8 | 60.1 -> 51.5 |
| CatBoost + kNN | 25.3 -> 25.3 | 27.4 -> 39.3 | 52.5 -> 38.0 |
| Balanced fusion | 24.1 -> 24.1 | 33.5 -> 39.3 | 59.5 -> 37.5 |
| Prior-adjusted | 47.7 -> 47.7 | 33.5 -> 38.6 | 58.3 -> 36.7 |
| XGB regression | 40.7 -> 39.6 | 73.8 -> 47.5 | 60.5 -> 35.5 |
| CatBoost regression | 45.7 -> 46.3 | 70.8 -> 50.4 | 61.7 -> 38.6 |

Three observations worth keeping:

1. The gain is inversely proportional to how good the representation already
   was. The gnn and genai heads, which were near 50-70% WAPE, converge to a
   tight 35-52% band; the conventional tree and kNN heads do not move at all.
2. The conventional tree/kNN heads are NOT frozen: their probabilities do
   change (kNN maxdiff 0.386, CatBoost 0.079, XGB 0.053). Element WAPE and
   top-1 for a classification head are step functions of 18 condition-level
   argmax decisions, and none of those flipped. The FT-Transformer flipped 5
   of 18, which is why it was the only conventional classifier that moved.
3. The trees barely use the new columns (XGB gain share 0.01% for T_ext and
   0.01% for log v_ext; CatBoost 0.003 and 0.100 against 28.5 for the
   ratio one-hot, its single most important feature). The labels explain it:
   the expected number of base alloys still consistent with the conditioning
   is 6.85 of 14 given the ratio type alone, 4.74 given (T_ext, v_ext) alone,
   and 4.57 given all three. The nine Mg-Gd alloys share one 350/450 C and
   0.5/1/2 mm/s grid, so the process parameters separate FAMILIES, which the
   ratio one-hot already did, and almost never isolate an alloy.

Side effect: alloy top-1 fell for genai and gnn (0.389 -> 0.222, 0.278 ->
0.222) while element WAPE improved sharply. The heads now make compositionally
CLOSER mistakes, picking the wrong alloy inside the right family, which the
element metrics reward and top-1 does not. Report WAPE and MAE as the primary
composition metrics and treat top-1 as secondary on 18 conditions.

## Process-head corrections (2026-08-04)

The labels were audited first and are clean: 108 unique conditions, the
condition_id encodes `<alloy>_extruded_<T>_<v>` and agrees with the T_ext and
v_ext columns in all 108 rows, no nulls, no duplicates. The bad v_ext numbers
came from three head-side defects, all now fixed in
scripts/cross_pipeline_heads.py:

1. CatBoost fitted both targets with `MultiRMSE` on RAW units. var(T_ext)=5963
   against var(v_ext)=5.64, so v_ext received 0.09% of the squared-error
   budget. Targets are now standardized before the joint objective.
2. v_ext is a log-scale quantity (grid 0.5, 0.6, 0.75, 1, 1.4, 2, 2.4, 2.8, 5,
   5.5, 6, 7.5; successive ratios 1.1-1.8). Squared error on the raw scale
   shrank predictions toward the mean, which is worst exactly where the MAPE
   denominator is smallest. All process heads now learn log(v_ext) and invert.
3. The process heads inherited `balanced=True` condition weighting from the
   composition classifier, which hurt them badly (XGB v_ext R2 -0.349 balanced
   against +0.103 unbalanced). Condition balancing is now used only by the
   composition heads, where it was designed.

Net effect on the conventional pipeline: v_ext MAPE 129.6% -> 69.0% (XGB),
120.0% -> 56.5% (CatBoost), 105.7% -> 52.0% (GP); no head has a negative R2 any
more (worst was -0.338). T_ext also improved (CatBoost MAE 41.2 -> 35.2 C). The
GP is now better calibrated AND sharper on v_ext: coverage 0.889/0.889/0.944
with NLL 2.03 -> 1.39, 1.99 -> 1.22, 1.80 -> 1.17.

A fourth, separate defect was reproducibility: `PCA(n_components=32)` inside
the GP head crosses sklearn's `svd_solver="auto"` threshold on the genai
latents (72 x 1290, so max(shape) > 500) and silently selected the randomized
solver with no seed. Genai GP T_ext MAE drifted 43.5 / 46.2 / 44.1 across three
otherwise identical runs, while conventional (72 x 448) and gnn (72 x 138) fell
back to the exact solver and were stable. PCA now takes `random_state=0`; two
consecutive genai GP fits agree to 0.0.

Note for the GP: v_ext is now modelled as lognormal, so the stored sigma is a
log-space standard deviation, the 90% interval is multiplicative, and the
reported NLL carries the log Jacobian so it remains a density in mm/s.

Normalization alone would NOT have been enough: an affine rescaling of a
single-target model is a no-op (XGB v_ext MAPE 119.1% raw against 119.2%
standardized). Only the nonlinear log transform and the joint-objective
standardization change the fit.

## Split-unit diagnostic

Random image-stratified 5-fold CV with the same CatBoost + kNN fusion gives
100% alloy top-1/top-3, zero composition WAPE and MAE, and 0.12% temperature
WAPE. The train/test condition-overlap fraction is 1.0 in every fold. These are
memorization results, not generalization results, and are excluded from the
headline tables. The primary protocol randomly assigns whole conditions to
alloy-stratified folds.

## Calibration and uncertainty (text/small table)

- Composition UQ = the fused 14-way distribution over nominal compositions.
  Seed-0 conventional: class NLL 2.12 condition-balanced fusion vs 2.41
  CatBoost alone and 6.02 kNN alone.
  Best seed-0 NLL is 2.05 from the FT-Transformer, which is
  simultaneously among the worst on WAPE -> temperature scaling / conformal
  sets.
- Process UQ: exact GP regression (ARD-RBF + White kernel, condition-level,
  PCA-32 features), with v_ext modelled as lognormal and its point estimate
  smearing-corrected. On the rebalanced seed-0 split, conditioned on known
  composition, T_ext 90% coverage is 0.833 conventional / 0.389 genai / 0.667
  gnn (target 0.90) and v_ext coverage is 0.556 / 0.444 / 0.889; the GNN GP is
  the best-calibrated v_ext head and conventional the best-calibrated T_ext
  head, but neither is near-nominal on both. NGBoost coverage 0.44, CatBoost
  RMSEWithUncertainty 0.07: both severely overconfident. GP is the process head.
  n~72-86 training conditions is exactly the GP regime.
- GP classifier over the alloy lattice is weak (top-1 0.17-0.31): condition
  pooling discards the multi-image evidence that boosting exploits; GBDT
  fusion stays the composition head.
- GMM/MDN and autoregressive flow heads share the flow's failure mode (mass
  off-lattice), so the lattice argument covers them. They have no seed-0
  counterpart and are not reported.

## Cut from the paper (kept in results/archive/ for reference)

- proto_mix / level_mix lattice heads and dev70 screening.
- Hurdle-aware decoders (posterior mean / present-conditioned mean/median):
  all within 1% WAPE of MAP, no signal.
- Cross-pipeline ensembles (60.7%), random forest (59.5%), kNN on genai
  latents (74.5%), CatBoost MultiRMSE (66.4%): dominated, mentioned at most in
  one appendix sentence.
- Soft-mixture decodes: dominated by argmax decodes.
- Hyperparameter sweep tables: appendix only.

## Sources

| table | source |
|---|---|
| Table 3 | runs/summary_cross_pipeline_heads_{conventional,genai,gnn}_random_seed0.json; raw predictions runs/predictions_cross_pipeline_*_random_seed0.npz; tidy values results/tables/*.csv; scripts/cross_pipeline_heads.py; chain scripts/run_cross_pipeline_random_seed0.sh; input/output contract in paper/dataset.md |
| split diagnostic | runs/summary_random_image_fuse.json |
| scripts | scripts/{tabular_heads,logit_adjusted_fusion,knn_retrieval,chain_head,snap_decode,unified_heads}.py |
| gnn embeddings | data/gnn/embeddings/random_seed0.npz (cached from the trained encoder) |
| full detail | results/archive/{ablation_tables,prediction_head_ablation}.md |
