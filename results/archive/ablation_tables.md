# Structure-to-Recipe: Full Results and Ablation Tables

All runs completed July 19, 2026. Three representation pipelines (gnn, conventional, genai) share the identical FlowTransformerHead (10 label tokens, 4 decoder blocks, dim 128, per-token rational-quadratic spline flows, hurdle presence logits, auxiliary 14-way alloy classifier). Only the encoder/adapter differs:

| pipeline | encoder | adapter tokens | trainable encoder |
|---|---|---|---|
| gnn | 4x GATv2 (dim 128, 4 heads, edge_dim 2) on grain graphs (90 um scale) | 8 attention-pool tokens | yes |
| conventional | 438-D handcrafted descriptor vector (4 scales) | 8 tokens (linear projection) | n/a (fixed features) |
| genai | frozen ViT-FMDiT latents (16 x 80) | 16 tokens (linear projection) | no (frozen) |

Data: 4348-image parity corpus (90 um: 1828, 100: 1261, 120: 751, 150: 508), 14 alloys.
Metrics: joint NLL (exact, lower better), macro presence F1, alloy top-1/top-3 (aux head), macro quantile ECE (lower better), 90% central-interval coverage (target 0.90), macro element MAE (wt.%).

---

## 1. Hyperparameter ablation (dev70 split, best validation joint NLL)

Base config for all: 400 epochs max with early stopping, batch by condition, samples-per-condition (spc) 16, AdamW.

### 1.1 Conventional pipeline

| config | lr | weight decay | spc | best val NLL | best epoch |
|---|---|---|---|---|---|
| **lr1e4 (winner)** | **1e-4** | **1e-4** | **16** | **1.789** | 123 |
| spc32 | 3e-4 | 1e-4 | 32 | 3.170 | 52 |
| wd1e3 | 3e-4 | 1e-3 | 16 | 3.545 | 94 |
| lr1e3wd1e3 | 1e-3 | 1e-3 | 16 | 4.001 | 77 |
| lr1e3 | 1e-3 | 1e-4 | 16 | 4.298 | 72 |

### 1.2 Genai pipeline

| config | lr | weight decay | spc | best val NLL | best epoch |
|---|---|---|---|---|---|
| **lr1e4 (winner)** | **1e-4** | **1e-4** | **16** | **-1.160** | 187 |
| spc32 | 3e-4 | 1e-4 | 32 | 0.580 | 66 |
| wd1e3 | 3e-4 | 1e-3 | 16 | 0.996 | 126 |
| lr1e3 | 1e-3 | 1e-4 | 16 | 2.592 | 95 |
| lr1e3wd1e3 | 1e-3 | 1e-3 | 16 | 2.735 | 93 |

Lower learning rate (1e-4) wins decisively for both adapter pipelines; extra weight decay and larger sampling budgets hurt. Winner configs were used for all cross-validation runs below.

### 1.3 GNN pipeline (dev70, 90 um graphs, lr 3e-4, wd 1e-4, spc 16)

| run | augmentation | best val NLL | best epoch |
|---|---|---|---|
| run4a | node/edge dropout 0.05 | 0.789 | 136 |
| **run5 (locked)** | **none (dropout 0)** | **0.828** | 136 |
| e400 base | none | 1.010 | 138 |
| run4b | dropout 0.05 + wd 1e-3 | 1.130 | 131 |
| flow_ar first version | none | 1.179 | 199 |
| run3 | node/edge dropout 0.10 | 1.204 | 135 |

Graph augmentation gives no reliable gain; the locked config uses no dropout. GNN CV runs use lr 3e-4, wd 1e-4, spc 16, 400 epochs, 90 um scale.

---

## 2. Main results: LOCO (leave-one-condition-group-out, 5 folds, mean +/- 95% CI)

| pipeline | joint NLL | pres F1 | top-1 | top-3 | ECE | cov90 | el. MAE |
|---|---|---|---|---|---|---|---|
| gnn | 0.553 +/- 0.479 | 0.394 +/- 0.162 | 0.16 +/- 0.10 | 0.28 +/- 0.06 | 0.320 +/- 0.018 | 0.23 +/- 0.04 | 0.709 +/- 0.107 |
| conventional | 2.460 +/- 2.826 | **0.648 +/- 0.132** | **0.31 +/- 0.12** | **0.46 +/- 0.09** | **0.299 +/- 0.014** | **0.37 +/- 0.13** | **0.478 +/- 0.152** |
| genai | **-1.150 +/- 1.744** | 0.553 +/- 0.073 | 0.20 +/- 0.06 | 0.34 +/- 0.07 | 0.322 +/- 0.021 | 0.18 +/- 0.06 | 0.550 +/- 0.086 |

Genai achieves the best density modeling (joint NLL); conventional descriptors win every discrete-decision and calibration metric; the GNN sits between on NLL despite seeing only the 90 um scale (single-scale caveat). All pipelines are overconfident (cov90 well below 0.90).

### 2.1 LOCO per-fold detail

| pipeline | fold | joint NLL | pres F1 | top-1 | top-3 | ECE | cov90 |
|---|---|---|---|---|---|---|---|
| gnn | fold0 | 0.561 | 0.546 | 0.25 | 0.35 | 0.307 | 0.190 |
| gnn | fold1 | 0.056 | 0.354 | 0.11 | 0.26 | 0.324 | 0.215 |
| gnn | fold2 | 0.308 | 0.208 | 0.05 | 0.23 | 0.301 | 0.253 |
| gnn | fold3 | 0.825 | 0.378 | 0.17 | 0.26 | 0.329 | 0.265 |
| gnn | fold4 | 1.015 | 0.485 | 0.22 | 0.30 | 0.336 | 0.251 |
| conventional | fold0 | 5.673 | 0.521 | 0.25 | 0.40 | 0.303 | 0.205 |
| conventional | fold1 | -0.173 | 0.760 | 0.42 | 0.58 | 0.295 | 0.445 |
| conventional | fold2 | 0.797 | 0.752 | 0.41 | 0.45 | 0.291 | 0.460 |
| conventional | fold3 | 3.235 | 0.576 | 0.22 | 0.39 | 0.289 | 0.353 |
| conventional | fold4 | 2.770 | 0.633 | 0.26 | 0.48 | 0.316 | 0.399 |
| genai | fold0 | 0.714 | 0.550 | 0.20 | 0.35 | 0.309 | 0.150 |
| genai | fold1 | -3.172 | 0.561 | 0.26 | 0.42 | 0.321 | 0.222 |
| genai | fold2 | -1.559 | 0.626 | 0.23 | 0.32 | 0.305 | 0.222 |
| genai | fold3 | -0.870 | 0.568 | 0.17 | 0.35 | 0.349 | 0.116 |
| genai | fold4 | -0.863 | 0.462 | 0.13 | 0.26 | 0.323 | 0.203 |

### 2.2 LOCO per-label MAE and WAPE (mean over 5 folds)

Two MAE variants are reported. "All-conditions" MAE averages over every test condition (absent elements count as true 0, matching the aggregate el. MAE in Section 2). "Present-only" MAE and WAPE restrict each element to conditions where it is truly present (> 0 wt.%). Percentage errors are reported as WAPE (weighted absolute percentage error, sum of absolute errors divided by sum of true values); point-wise MAPE is not used because trace-level true values (0.01-0.1 wt.%) make per-point ratios explode into the thousands of percent. WAPE is bounded and interpretable: 100% means the error mass equals the true mass (e.g. predicting the element absent). Elements in wt.%, T_ext in C, v_ext in mm/s.

All-conditions MAE:

| pipeline | Al | Zn | Mn | Ce | Gd | Ca | Nd | Y | T_ext | v_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 1.895 | 0.474 | 0.479 | 0.096 | 2.603 | 0.015 | 0.106 | 0.005 | 68.9 | 1.81 |
| conventional | **0.270** | **0.343** | **0.263** | **0.032** | 2.708 | 0.015 | 0.188 | **0.002** | **55.3** | 1.90 |
| genai | 1.147 | 0.360 | 0.482 | 0.105 | **2.168** | 0.015 | 0.119 | 0.006 | 63.2 | **1.80** |

Present-only MAE:

| pipeline | Al | Zn | Mn | Ce | Gd | Ca | Nd | Y | T_ext | v_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 1.283 | 0.268 | 0.671 | 0.392 | 5.035 | 0.150 | 0.227 | 0.022 | 68.9 | 1.81 |
| conventional | **0.534** | **0.262** | **0.283** | **0.066** | 4.225 | 0.149 | **0.117** | **0.003** | **55.3** | 1.90 |
| genai | 0.778 | 0.276 | 0.417 | 0.134 | **3.755** | 0.150 | 0.211 | 0.008 | 63.2 | **1.80** |

Present-only WAPE (%):

| pipeline | Al | Zn | Mn | Ce | Gd | Ca | Nd | Y | T_ext | v_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 172.7 | 42.4 | 57.2 | 56.0 | 88.5 | 100.0 | 100.7 | 55.8 | 18.5 | **70.7** |
| conventional | **51.6** | **40.4** | **25.2** | **9.4** | 74.7 | 99.2 | **50.9** | **6.7** | **14.8** | 79.3 |
| genai | 84.6 | 43.1 | 37.0 | 19.2 | **66.2** | 100.0 | 93.5 | 19.2 | 17.0 | 71.2 |

Gd (the widest-ranging element, 0-10 wt.%) dominates the macro MAE for all pipelines. Ca at ~100% WAPE is effectively never recovered (present at only 0.15 wt.% in a single alloy family). GNN Al above 100% reflects systematic over-prediction of Al mass. Process parameters are the best-predicted targets in relative terms (T_ext WAPE 15-19%).

---

## 3. LOAO (leave-one-alloy-out, 14 folds)

### 3.1 The out-of-support effect

Three held-out alloys contain an element that appears in no other alloy of the corpus: AZ31 (only Al-bearing alloy), ME21 (only Ce), ZNd10 (only Nd). Under LOAO the flows have never seen these elements present, assign near-zero density, and the joint NLL diverges. This happens identically for all three pipelines, i.e. it is a property of the task (novel chemistry is out of support), not of any representation. Raw 14-fold means are therefore dominated by these three folds and are reported only for completeness.

### 3.2 In-support LOAO (11 folds, novel-chemistry alloys excluded; mean +/- 95% CI)

| pipeline | joint NLL (in-support) | median NLL (all 14) | pres F1 (in-support) |
|---|---|---|---|
| gnn | 8.398 +/- 1.978 | 10.61 | 0.092 |
| conventional | 9.261 +/- 2.012 | 10.90 | 0.153 |
| genai | **7.726 +/- 2.045** | 10.44 | 0.132 |

### 3.3 Full 14-fold LOAO summary (raw means, dominated by out-of-support folds)

| pipeline | joint NLL | pres F1 | top-1 | top-3 | ECE | el. MAE |
|---|---|---|---|---|---|---|
| gnn | 272313 +/- 394667 | 0.092 +/- 0.034 | 0.00 | 0.15 +/- 0.20 | 0.418 +/- 0.024 | 0.942 +/- 0.262 |
| conventional | 495668 +/- 637212 | 0.144 +/- 0.035 | 0.07 +/- 0.07 | 0.20 +/- 0.14 | 0.371 +/- 0.038 | 0.695 +/- 0.177 |
| genai | 289504 +/- 389430 | 0.144 +/- 0.046 | 0.02 +/- 0.04 | 0.19 +/- 0.18 | 0.420 +/- 0.020 | 0.732 +/- 0.182 |

(cov90 is undefined for LOAO folds because held-out composition labels are constant per fold.)

### 3.4 LOAO per-fold detail

Out-of-support folds marked with *.

| held-out alloy | gnn NLL | conv NLL | genai NLL | gnn F1 | conv F1 | genai F1 |
|---|---|---|---|---|---|---|
| AZ31 * | 2345818 | 3733722 | 2180698 | 0.125 | 0.152 | 0.338 |
| ME21 * | 216481 | 1914995 | 398027 | 0.025 | 0.118 | 0.114 |
| ZNd10 * | 1249984 | 1290538 | 1474252 | 0.119 | 0.067 | 0.113 |
| Mg-2Gd | 10.92 | 11.67 | 11.36 | 0.000 | 0.077 | 0.025 |
| Mg-2Gd-0.5Mn | 5.65 | 7.72 | 4.62 | 0.125 | 0.232 | 0.175 |
| Mg-2Gd-1Mn | 10.52 | 10.12 | 9.66 | 0.107 | 0.232 | 0.208 |
| Mg-5Gd | 7.47 | 8.64 | 8.31 | 0.000 | 0.117 | 0.068 |
| Mg-5Gd-0.5Mn | 4.00 | 4.11 | 2.15 | 0.125 | 0.214 | 0.125 |
| Mg-5Gd-1Mn | 5.23 | 4.71 | 4.32 | 0.175 | 0.190 | 0.208 |
| Mg-10Gd | 10.71 | 12.39 | 11.23 | 0.000 | 0.077 | 0.045 |
| Mg-10Gd-0.5Mn | 6.38 | 5.97 | 6.33 | 0.107 | 0.190 | 0.175 |
| Mg-10Gd-1Mn | 5.57 | 9.27 | 4.45 | 0.125 | 0.167 | 0.175 |
| Z1 | 11.50 | 13.20 | 11.32 | 0.125 | 0.089 | 0.125 |
| ZX10 | 14.43 | 14.08 | 11.22 | 0.125 | 0.097 | 0.125 |

### 3.5 LOAO per-label MAE and WAPE (mean over 14 folds)

Present-only MAE (elements masked to conditions where truly present; process labels over all conditions):

| pipeline | Al | Zn | Mn | Ce | Gd | Ca | Nd | Y | T_ext | v_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 1.445 | **0.089** | 0.603 | 0.700 | 5.354 | 0.150 | 0.292 | 0.040 | 55.8 | **1.62** |
| conventional | 1.445 | 0.414 | **0.497** | 0.700 | 4.333 | 0.150 | 0.292 | 0.040 | 53.0 | 1.71 |
| genai | 1.544 | 0.298 | 0.584 | 0.700 | **3.610** | 0.150 | 0.292 | 0.040 | **50.4** | 1.65 |

Present-only WAPE (%):

| pipeline | Al | Zn | Mn | Ce | Gd | Ca | Nd | Y | T_ext | v_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn | **100.0** | **29.1** | 65.6 | 100.0 | 93.3 | 100.0 | 100.0 | 100.0 | 15.5 | **83.6** |
| conventional | **100.0** | 138.7 | **61.0** | 100.0 | 90.8 | 100.0 | 100.0 | 100.0 | **14.5** | 108.1 |
| genai | 1114.9 | 119.3 | 105.5 | 100.0 | **57.2** | 100.0 | 100.0 | 100.0 | 14.6 | 90.4 |

Under LOAO, Al, Ce, Nd, Y, and Ca sit at exactly 100% WAPE for pipelines that predict them absent: each of these elements is present only in the held-out alloy, so the model predicts ~0 and the error mass equals the true mass. This is the tabular counterpart of the out-of-support NLL divergence in Section 3.1 (genai Al above 100% means it hallucinates Al content on AZ31 rather than predicting absence). Only Zn, Mn, Gd, and the process parameters carry meaningful LOAO relative errors; extrusion temperature remains well predicted (WAPE ~15%) even for unseen alloys.

---

## 4. Prediction-head ablation and lattice-snapped decoding

The corpus contains only 14 distinct alloy compositions; element amounts live on a small discrete lattice (e.g. Gd in {0, 2, 5, 10} wt.%). The continuous flow head spreads probability mass between lattice points, which inflates element MAE. Two remedies were tested: (a) new lattice-aware heads, and (b) a post-hoc decoding rule that snaps the posterior mean of the composition samples to the nearest training alloy's nominal composition (z-scored element distance). Process labels always keep the posterior mean.

### 4.1 Head screening (dev70, conventional pipeline; raw MAP decoding)

| head | joint NLL | presence F1 | alloy top-1 | alloy top-3 | element MAE | Gd MAE |
|---|---|---|---|---|---|---|
| flow_ar (baseline) | **1.816** | 0.700 | 0.33 | 0.58 | **0.410** | 2.297 |
| level_mix lr 3e-4 | 4.333 | 0.734 | **0.38** | **0.71** | 0.414 | **2.094** |
| level_mix lr 1e-4 | 4.328 | **0.760** | 0.33 | 0.54 | 0.475 | 2.545 |
| level_mix (attn pool) lr 3e-4 | 4.644 | 0.694 | 0.38 | 0.62 | 0.428 | 2.180 |
| proto_mix (all variants) | collapsed | ~0.34 | 0.17-0.21 | -- | 0.64-0.68 | -- |

proto_mix (mixture over full alloy prototypes) collapses: the hard coupling of all 8 elements to one prototype gives no partial credit and training degenerates. level_mix (per-element hurdle + categorical over observed levels) is competitive and wins the discrete decision metrics, but its NLL is not comparable in kind to the flow's continuous density (mixture of narrow Gaussians at levels). Mean pooling beats attention pooling; lr 3e-4 is the winner config.

### 4.2 Lattice-snapped decoding (existing flow_ar runs, no retraining)

Point estimate = posterior mean over 256 samples, elements snapped to the nearest training alloy nominal. For LOAO the candidate set excludes the held-out alloy. LOCO (5 folds):

| pipeline | element MAE raw -> snap | Gd MAE raw -> snap | presence F1 raw -> snap | alloy top-1 | alloy top-3 |
|---|---|---|---|---|---|
| gnn | 0.709 -> **0.499** | 2.603 -> 2.826 | 0.648* -> 0.266 | 0.10 | 0.27 |
| conventional | 0.478 -> **0.412** | 2.708 -> **2.505** | 0.648 -> **0.728** | 0.30 | 0.49 |
| genai | 0.550 -> **0.391** | 2.168 -> 2.201 | 0.553 -> **0.635** | 0.25 | 0.37 |

LOAO (14 folds):

| pipeline | element MAE raw -> snap | presence F1 raw -> snap | top-1 raw -> snap | top-3 raw -> snap |
|---|---|---|---|---|
| gnn | 0.942 -> **0.568** | 0.092 -> 0.121 | 0.00 -> 0.03 | 0.15 -> 0.15 |
| conventional | 0.695 -> **0.550** | 0.144 -> 0.150 | 0.07 -> 0.11 | 0.20 -> 0.27 |
| genai | 0.732 -> **0.547** | 0.144 -> 0.149 | 0.02 -> 0.06 | 0.14 -> 0.14 |

*GNN raw presence F1 is from sampled presence; its snapped F1 (0.266) is poor because the GNN posterior means sit far off-grid, so snapping picks wrong alloys. Snapping helps only when the underlying representation is decent.

**WAPE caveat.** Present-only element WAPE does NOT improve with snapping: LOCO raw -> snap is 84.2 -> 85.1% (gnn), 44.8 -> 45.1% (conventional), 57.8 -> 63.4% (genai); LOAO worsens sharply (72.7 -> 87.5%, 81.1 -> 118.3%, 96.0 -> 158.0%). The MAE gain comes from zeroing spurious trace mass on the many absent elements, while snapping to a wrong alloy sets truly present elements to 0 or a wrong level (all-or-nothing relative error). Under LOAO the true alloy is excluded from the candidate set, so the snap is guaranteed wrong on the held-out alloy's signature elements. Consequently snapped point estimates are reported for LOCO only; raw MAP remains the estimator for present-element relative error and for all LOAO tables. T_ext WAPE improves slightly with the posterior mean (to ~14-15%) in all settings.

### 4.3 Combined: level_mix + snap on conventional LOCO (5 folds)

| variant | joint NLL | presence F1 | top-1 | top-3 | element MAE | Gd MAE |
|---|---|---|---|---|---|---|
| flow_ar, raw MAP | **2.460** | 0.648 | 0.31 | 0.46 | 0.478 | 2.708 |
| flow_ar + snap | -- | 0.728 | 0.30 | 0.49 | 0.412 | 2.505 |
| level_mix, raw MAP | 4.981 | 0.698 | 0.32 | **0.50** | 0.460 | **2.579** |
| level_mix + snap | -- | **0.751** | **0.33** | 0.49 | **0.358** | -- |

**Chosen configuration:** keep flow_ar as the density model for NLL/calibration reporting (its continuous likelihood is the honest generative metric), and report point estimates with lattice-snapped decoding (posterior mean + snap-to-training-alloy). level_mix + snap gives the best decision metrics (element MAE 0.358, presence F1 0.751) and is reported as the head-ablation row. Element error is dominated by alloy identification: Gd MAE ~2-2.6 wt.% equals the lattice spacing (2/5/10), so with 14 discrete alloys the composition task is effectively a ~3.8-bit classification problem, not regression.

Sources: `runs/head_*` (dev70 screening), `runs/conventional_levelmix_loco_*`, `runs/summary_snap_{loco,loao}_{gnn,conventional,genai}.json`, per-run `snap_eval_*.json` via `scripts/snap_decode.py`.

### 4.4 Retrieval decoding (kNN): the scarce-data winner

Nonparametric retrieval over the cached conventional features (438-D, z-scored, cosine similarity, k=5 image neighbors, temperature 0.05, per-condition pooling over all test images of a condition; prediction = similarity-weighted label average). No training. Script: `scripts/knn_retrieval.py`.

LOCO comparison of all point-prediction strategies (5 folds):

| method | element WAPE (present) | element MAE | alloy top-1 | alloy top-3 | T_ext WAPE |
|---|---|---|---|---|---|
| flow_ar raw MAP (conventional) | 44.8% | 0.478 | 0.31 | 0.46 | 14.8% |
| flow_ar + snap decode | 45.1% | 0.412 | 0.30 | 0.49 | 14.0% |
| level_mix + snap | -- | 0.358 | 0.33 | 0.49 | -- |
| cross-pipeline MAP ensemble | 60.7% | 0.548 | -- | -- | -- |
| random forest (438-D features) | 59.5% | 0.477 | -- | -- | 13.5% |
| kNN on genai latents | 74.5% | 0.507 | -- | -- | 18.1% |
| **kNN retrieval (conventional)** | **34.3%** | **0.374** | **0.48** | **0.61** | **13.1%** |

Per-element kNN LOCO: present-only WAPE Al 24.3%, Zn 34.4%, Mn 25.4%, Ce 2.7%, Gd 60.7%, Ca 39.6%, Nd 84.6%, Y 2.7%; T_ext 13.1%, v_ext 57.1%. Gd remains the hardest (lattice spacing 2/5/10 wt.%).

LOAO: kNN element WAPE 118% with alloy top-1 = 0 by construction — retrieval cannot return an alloy absent from the training set. This is the honest nonparametric statement of the LOAO identifiability limit: no method that decodes to seen compositions can generalize to unseen chemistry, and the parametric flows do not truly do better (their LOAO WAPE is 81-96% with signature elements at exactly 100%).

**Final recipe for the paper (point prediction):** kNN retrieval decoding for LOCO point estimates (element WAPE 44.8 -> 34.3%, alloy top-1 0.31 -> 0.48, T_ext WAPE 13.1%); flow_ar retained as the probabilistic model for densities, calibration, and sampling. Framing: in the scarce-data regime (14 alloys, ~100 conditions) an amortized inverse model cannot beat retrieval on point accuracy; its value is uncertainty quantification, while retrieval supplies the recipe.

### 4.5 Tabular model bake-off: GBDT, FT-Transformer, and probabilistic classification

Literature grounding: Grinsztajn et al. (NeurIPS 2022) show gradient-boosted trees dominate deep nets on small/medium tabular data; FT-Transformer (Gorishniy et al., NeurIPS 2021) is the strongest deep tabular baseline; NGBoost (Duan et al., ICML 2020) and CatBoost RMSEWithUncertainty provide distributional regression. For this task the composition label space is 14 discrete alloys, so the theoretically right uncertainty object for composition is a **probability distribution over nominal compositions** (a 14-way classifier), not a continuous density. Script: `scripts/tabular_heads.py` (image-level training on conventional features, per-condition aggregation of predictions/probabilities, same protocol as kNN).

LOCO (5 folds), composition decoded from the classifier argmax nominal, T/v from XGBoost regression (except FTT which regresses its own):

| model | element WAPE (present) | element MAE | alloy top-1 | alloy top-3 | T_ext WAPE |
|---|---|---|---|---|---|
| flow_ar raw MAP (reference) | 44.8% | 0.478 | 0.31 | 0.46 | 14.8% |
| kNN retrieval (Section 4.4) | 34.3% | 0.374 | 0.48 | 0.61 | 13.1% |
| XGBoost regression (per target) | 35.3% | 0.414 | 0.42 | 0.54 | 13.1% |
| XGBoost 14-way classifier | 36.6% | 0.346 | 0.40 | 0.66 | 13.1% |
| CatBoost MultiRMSE regression | 66.4% | 0.484 | 0.24 | 0.37 | 14.1% |
| CatBoost 14-way classifier | 25.7% | 0.322 | 0.50 | 0.68 | 13.1% |
| FT-Transformer (cls + reg) | 30.2% | 0.396 | 0.46 | 0.62 | **12.1%** |
| **CatBoost + kNN probability fusion** | **21.7%** | **0.292** | **0.54** | **0.68** | 13.1% |

The winner is a 50/50 fusion of CatBoost classifier probabilities with the kNN neighbor-vote distribution (`fuse_argmax`): element WAPE halves versus the flow head (44.8 -> 21.7%), alloy top-1 nearly doubles (0.31 -> 0.54). The fused distribution is also the best-calibrated classifier: per-condition class NLL 1.66 vs 1.95 (CatBoost) and 5.08 (kNN); mean |confidence - accuracy| 0.32 (still overconfident; temperature scaling or conformal sets are the fix). Regression-style heads (CatBoost MultiRMSE, XGB per-target) confirm the diagnosis that composition should be classified, not regressed.

Uncertainty for process parameters (LOCO, per-condition): NGBoost Normal gives T_ext NLL 6.71 with 90% interval coverage 0.44; CatBoost RMSEWithUncertainty is badly overconfident (coverage 0.07). Neither is usable without recalibration; split-conformal intervals on top of the point regressor are the pragmatic choice. FT-Transformer is the best T_ext point predictor (12.1% WAPE).

**Revised final recipe:** composition = fused CatBoost + kNN probability distribution over the 14 nominal compositions (decode argmax for the point recipe, report the full distribution as the uncertainty statement); process parameters = XGBoost/FTT point estimate with split-conformal intervals; flow_ar retained for joint-density comparison across the three representation pipelines. Sources: `runs/summary_tabular_*_loco.json`.

The complete consolidated head/decoder ablation (all 19 strategies, calibration, UQ, LOAO, and reproducibility details) is in `results/prediction_head_ablation.md`.

---

## 5. Notes for the paper

- LOCO is the primary table; LOAO reported as in-support (11 folds) vs out-of-support (3 folds), with the divergence discussed as an identifiability limit of inverse chemistry prediction.
- Complementary strengths: genai (frozen generative latents) best at density modeling; conventional descriptors best at discrete decisions and calibration; GNN competitive on NLL from a single scale only.
- All pipelines are overconfident (LOCO cov90 0.18-0.37 vs target 0.90) and ECE ~ 0.30; conformal or temperature recalibration is future work.
- GNN caveat: graphs available at 90 um scale only, while conventional uses all 4 scales and genai latents are scale-agnostic.
- Sources: `runs/summary_loco.json` (gnn), `runs/summary_loco_conventional.json`, `runs/summary_loco_genai.json`, `runs/summary_loao_{gnn,conventional,genai}.json`, per-fold `runs/*_lo{co,ao}_*/eval_*.json`, sweep `runs/sweep_*/result.json`, winners `runs/sweep_winners.json`. Present-only MAE/WAPE computed from MAP predictions in `runs/*_lo{co,ao}_*/predictions_*.npz` (y_map vs y_true, elements masked to true > 0; WAPE = sum|err| / sum|true| per fold, averaged over folds).
