# Dataset and Prediction Tasks

## Dataset unit and protocol

The canonical label table is `data/labels/labels.csv`. It contains 108 unique
processing conditions from 14 base alloys. Each image or graph is linked to one
condition through `condition_id`; labels are constant within that condition.

The composition space is reduced to seven informative elements: `Al`, `Zn`,
`Mn`, `Ce`, `Gd`, `Ca`, and `Nd`. The source table `data_labels.xlsx` is
kept in the same reduced form. The trace columns it once carried (`Cu`, `Ni`,
`Si`, `Fe`, `Pr`) were dropped: `Cu` and `Ni` are constant ppm impurities that
turn on and off with the whole Mg-Gd family, `Fe` scales with the Gd source,
and `Si`/`Pr` appear only in ME21 at 0.01-0.015 wt%. `Y` is dropped for the
same reason as `Si` and `Pr` (it marks ME21 alone at 0.04 wt%), and with `Y`
gone the remaining seven elements still give all 14 alloys a unique nominal
signature. None of the dropped columns distinguishes one alloy from another,
and the retained seven reconstruct every nominal composition exactly.

The primary evaluation uses one frozen alloy-stratified random split over whole
conditions (`data/splits/random_seed0.json`). The random seed is 0. Images or
graphs from one condition never cross train, validation, and test partitions.
There is no leave-one-condition-out or leave-one-alloy-out evaluation in this
protocol. Predictions are aggregated to condition level before metrics are
calculated.

## Structure representations

The same two prediction tasks are evaluated with three alternative structure
representations:

| Pipeline | Per-sample representation | Current dimension |
|---|---|---:|
| Conventional | Reduced microstructure and texture descriptors | 438 |
| GenAI | Flattened `16 x 80` ViT-FMDiT latent tokens | 1,280 |
| GNN | Mean-pooled trained GATv2 graph tokens at 90 um | 128 |

The representation standardizer is fitted on the training partition and then
applied to validation and test data.

`AZ31_extruded_500_6` has no ODF/RVE-derived sample and is absent from the
training partition of ALL THREE representations, so every pipeline trains on 71
of the 72 assigned training conditions. It is not a validation or test
condition in `random_seed0`; validation and test coverage remains 18/18 for all
three pipelines. Every result artifact records expected, observed, and missing
condition coverage.

Sample counts per partition differ between pipelines because RVE graphs exist
for only a subset of the images:

| Pipeline | Train images | Val images | Test images |
|---|---:|---:|---:|
| Conventional | 2,909 | 729 | 710 |
| GenAI | 2,909 | 729 | 710 |
| GNN | 1,209 | 317 | 302 |

The GNN therefore sees roughly 42% of the per-image evidence available to the
other two pipelines at identical condition coverage. Metrics are aggregated per
condition, so the comparison remains condition-level fair, but this imbalance
should be kept in mind when reading composition results.

## Shared extrusion-ratio metadata

The available sources identify two extrusion-ratio categories but do not store
an authoritative physical ratio value. The canonical categorical field is
`extrusion_ratio_type`:

| Category | Base alloys |
|---|---|
| `mg_gd_series` | All Mg-Gd and Mg-Gd-Mn alloys |
| `standard` | AZ31, ME21, Z1, ZNd10, and ZX10 |

Models receive this field as a two-column one-hot encoding in the fixed order
`[standard, mg_gd_series]`. Both columns are retained to make the two known
processing routes explicit.

## Task A: alloy-composition characterization

This task estimates material identity and composition from observed structure
when the processing route is known. Task A and Task B are deliberately
symmetric: each is conditioned on the half of the condition label that the
other one predicts, so neither head has to disentangle composition effects from
processing effects on its own.

**Inputs**

- One of the three structure representations.
- Known extrusion parameters: `T_ext` and `v_ext`. `v_ext` enters as
  `log(v_ext)` because it spans 1.2 decades on a near-geometric grid, while
  `T_ext` stays linear over its 0.4 decades. Both are then standardized with
  training-partition statistics.
- Two-column extrusion-ratio encoding.

The resulting input dimensions are 442 for Conventional, 1,284 for GenAI, and
132 for GNN.

**Outputs**

- Seven element concentrations in wt%: `Al`, `Zn`, `Mn`, `Ce`, `Gd`, `Ca`,
  and `Nd`.
- Classification heads additionally predict one of the 14 base-alloy classes;
  its nominal composition is used for the element estimate.

Classification metrics are present-element WAPE, element MAE, alloy top-1,
alloy top-3, and class NLL. Direct regression heads report present-element WAPE
and element MAE. Per-element MAE and present-only WAPE are always retained.

Composition heads see no alloy identifier at input and are trained with
condition balancing, so every processing condition carries equal total weight
and alloys represented by only four conditions are not swamped.

## Task B: process-parameter characterization

This task estimates processing parameters after the material composition is
known. The known composition is therefore an input, not a prediction target or
test-label leak under this deployment definition. The same argument applies in
reverse to the known extrusion parameters in Task A.

**Inputs**

- One of the three structure representations.
- Known concentrations of the seven elements in the fixed order above.
- Two-column extrusion-ratio encoding.

The known composition is standardized using training-partition statistics.
The resulting input dimensions are 447 for Conventional, 1,289 for GenAI, and
137 for GNN.

**Outputs**

- `T_ext`: extrusion temperature in degrees Celsius.
- `v_ext`: extrusion ram velocity in mm/s. All heads learn `log(v_ext)` and
  invert the prediction; the Gaussian process therefore models `v_ext` as
  lognormal, so its interval is multiplicative and its NLL carries the log
  Jacobian.

Both outputs are evaluated with MAE, WAPE, MAPE, and R2. Gaussian-process heads
also report predictive NLL and empirical 90% interval coverage. The two targets
have very different MAPE floors, since a constant train-mean predictor scores
14.5% on `T_ext` and 136.5% on `v_ext`, so MAPE is comparable down a column and
never across the two targets.

Unlike the composition heads, the process heads are trained UNBALANCED.
Condition balancing was designed for the 14-class composition problem and
measurably hurts process regression.

## Evaluation notes

The test partition holds 18 conditions, so alloy top-1 moves in steps of 0.056
and head-to-head gaps below roughly 0.15 are not resolvable. Element WAPE and
top-1 for a classification head are step functions of those 18 condition-level
argmax decisions, which means a head can change its probabilities substantially
and still report an identical metric. Rankings inside one pipeline are treated
as noise; only cross-pipeline patterns are claimed.

Every run stores the raw per-image test predictions next to the aggregate
metrics, including the condition-level Gaussian-process mean and predictive
sigma, so a newly requested metric is a recomputation rather than a re-fit.

## Benchmark interpretation

Only heads trained with the task-specific conditioned inputs above appear in
the cross-pipeline comparison: kNN retrieval, XGBoost, CatBoost, an
FT-Transformer, fusion variants on the composition side, and an exact Gaussian
process on the process side. The earlier Flow-AR family has no seed-0
counterpart and is dropped rather than presented as a like-for-like conditioned
result.

Validation conditions select any data-dependent correction, including the
composition prior-adjustment exponent. Test conditions are used once, for the
reported split metrics.