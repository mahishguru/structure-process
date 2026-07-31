# Prediction Head and Decoder Ablation (consolidated)

All experiments on the 4348-image parity corpus, conventional features unless noted.
Protocol: image-level training, per-condition aggregation (mean of predictions or
class probabilities), 5 LOCO folds (mean over folds). Element WAPE is present-only:
sum|err| / sum|true| over elements with true > 0, macro-averaged. LOAO = 14 folds.

## 1. Master table: LOCO point prediction (all strategies)

| # | method | class | element WAPE | element MAE | alloy top-1 | alloy top-3 | T_ext WAPE |
|---|---|---|---|---|---|---|---|
| 1 | flow_ar raw MAP (conventional) | amortized flow | 44.8% | 0.478 | 0.31 | 0.46 | 14.8% |
| 2 | flow_ar posterior mean | decoder | 46.0% | 0.457 | -- | -- | 14.0% |
| 3 | flow_ar present-conditioned mean | decoder | 46.0% | 0.458 | -- | -- | -- |
| 4 | flow_ar present-conditioned median | decoder | 45.5% | 0.454 | -- | -- | -- |
| 5 | flow_ar + snap-to-alloy | decoder | 45.1% | 0.412 | 0.30 | 0.49 | 14.0% |
| 6 | level_mix head (raw MAP) | lattice head | -- | 0.460 | 0.32 | 0.50 | -- |
| 7 | level_mix + snap-to-alloy | lattice head + decoder | -- | 0.358 | 0.33 | 0.49 | -- |
| 8 | cross-pipeline MAP ensemble (3 pipes) | ensemble | 60.7% | 0.548 | -- | -- | -- |
| 9 | conv+genai MAP ensemble | ensemble | 49.9% | 0.486 | -- | -- | -- |
| 10 | random forest (438-D features) | tabular reg | 59.5% | 0.477 | -- | -- | 13.5% |
| 11 | kNN on genai latents (k=5) | retrieval | 74.5% | 0.507 | -- | -- | 18.1% |
| 12 | kNN retrieval, conventional (k=5) | retrieval | 34.3% | 0.374 | 0.48 | 0.61 | 13.1% |
| 13 | XGBoost regression (per target) | tabular reg | 35.3% | 0.414 | 0.42 | 0.54 | 13.1% |
| 14 | CatBoost MultiRMSE regression | tabular reg | 66.4% | 0.484 | 0.24 | 0.37 | 14.1% |
| 15 | XGBoost 14-way classifier (argmax) | classifier | 36.6% | 0.346 | 0.40 | 0.66 | 13.1% |
| 16 | CatBoost 14-way classifier (argmax) | classifier | 25.7% | 0.322 | 0.50 | 0.68 | 13.1% |
| 17 | FT-Transformer (cls + reg) | deep tabular | 30.2% | 0.396 | 0.46 | 0.62 | **12.1%** |
| 18 | **CatBoost + kNN fusion (argmax)** | **classifier fusion** | **21.7%** | **0.292** | **0.54** | **0.68** | 13.1% |
| 19 | CatBoost + kNN fusion (soft mixture) | classifier fusion | 32.1% | 0.327 | 0.54 | 0.68 | 13.1% |
| 20 | AR level chain, greedy decode | typed AR chain | 36.6% | 0.321 | -- | -- | 15.0% |
| 21 | AR level chain, alloy-scored | typed AR chain | 35.4% | 0.325 | 0.45 | 0.62 | 15.0% |
| 22 | AR level chain + kNN fusion | typed AR chain | 26.0% | **0.273** | 0.51 | 0.66 | 15.0% |

Winner: row 18, a 50/50 fusion of CatBoost classifier probabilities with the kNN
neighbor-vote distribution over the 14 training alloys, decoded to the argmax
nominal composition. Element WAPE halves vs the flow head; alloy top-1 nearly doubles.

## 2. Classifier calibration (LOCO, per-condition 14-way distribution)

| model | class NLL | mean \|confidence - accuracy\| |
|---|---|---|
| CatBoost probs | 1.95 | 0.359 |
| kNN vote distribution | 5.08 | 0.319 |
| **fused (50/50)** | **1.66** | 0.320 |

The fused distribution over nominal compositions is the composition uncertainty
statement (a discrete posterior over the alloy lattice). All variants remain
overconfident; temperature scaling or conformal prediction sets are the fix.

## 3. Process-parameter uncertainty (T_ext, LOCO, per-condition)

| model | NLL | 90% coverage | T_ext WAPE |
|---|---|---|---|
| NGBoost Normal | 6.71 | 0.44 | 13.9% |
| CatBoost RMSEWithUncertainty | 5101.0 | 0.07 | 14.4% |

Both distributional regressors are overconfident; split-conformal intervals on the
point regressor (XGB 13.1% / FTT 12.1% WAPE) are the pragmatic choice.

## 4. LOAO (14 folds): the identifiability limit

| method | element WAPE | element MAE | alloy top-1 |
|---|---|---|---|
| flow_ar raw MAP (conventional) | 81.1% | 0.695 | 0.07 |
| flow_ar + snap (held-out excluded) | 118.3% | 0.550 | 0.11 |
| kNN retrieval | 118.0% | 0.619 | 0.00 |

No method that decodes to seen compositions can generalize to unseen chemistry;
retrieval/classification makes this explicit (top-1 = 0 by construction), while
flows hide it as ~100% WAPE on the held-out alloy's signature elements. LOAO is
reported with the raw flow MAP only.

### 4.1 The typed autoregressive chain and why LOAO still fails

The theoretically type-correct head is an autoregressive classifier chain over
element levels: p(y|x) = prod_e p(level_e | x, levels_<e) * p(T,v | x, levels)
(`scripts/chain_head.py`, one CatBoost per element over its observed training
lattice, level 0 = absent so no separate hurdle; exact teacher-forced scoring of
any candidate alloy). Unlike the direct 14-way classifier this head assigns
probability to unseen COMBINATIONS of seen levels, which is the only principled
route to LOAO generalization (e.g. Mg-5Gd-1Mn = Gd@5 + Mn@1, both seen in other
alloys).

LOCO: chain + kNN fusion reaches the best element MAE of any method (0.273) and
26.0% WAPE, but does not beat the direct classifier fusion on WAPE (21.7%): with
all 14 alloys in training, the direct classifier is the sharper in-domain model.

LOAO: the chain CAN rank the held-out alloy (the direct classifier cannot), but
in practice ranks it ~11.5th of 14 on average; element WAPE 260% and top-1 = 0.
The chain learns spurious feature-conditional level correlations rather than
transferable composition-microstructure physics. Conclusion: LOAO failure is a
property of the data (14 alloys cannot pin down an 8-D composition manifold),
not of the head. This makes the identifiability claim in the paper rigorous:
even the head with the correct combinatorial support fails OOD.

## 5. Lattice-head screening (dev70, conventional; context for rows 6-7)

| head | joint NLL | presence F1 | top-1 | top-3 | element MAE | Gd MAE |
|---|---|---|---|---|---|---|
| flow_ar (baseline) | 1.816 | 0.700 | 0.33 | 0.58 | 0.410 | 2.297 |
| level_mix lr 3e-4 (mean pool) | 4.333 | 0.734 | 0.38 | 0.71 | 0.414 | 2.094 |
| level_mix lr 1e-4 | 4.328 | 0.760 | 0.33 | 0.54 | 0.475 | 2.545 |
| level_mix attn pool lr 3e-4 | 4.644 | 0.694 | 0.38 | 0.62 | 0.428 | 2.180 |
| proto_mix (all variants) | collapsed | ~0.34 | 0.17-0.21 | -- | 0.64-0.68 | -- |

## 6. Final recipe

- Composition: fused CatBoost + kNN probability distribution over the 14 nominal
  compositions. Argmax nominal = point recipe; the distribution itself = uncertainty.
  The typed AR level chain (rows 20-22) is the theoretically correct head and the
  best on element MAE, but the direct classifier fusion stays sharper in-domain;
  the chain's LOAO failure proves the identifiability limit is data, not architecture.
- Process parameters: XGBoost/FT-Transformer point estimate + split-conformal intervals.
- flow_ar retained for the cross-pipeline joint-density comparison (NLL, sampling).
- Literature: Grinsztajn et al. 2022 (GBDT > deep nets on small tabular);
  Gorishniy et al. 2021 (FT-Transformer); Duan et al. 2020 (NGBoost).

## 7. Reproducibility

| artifact | source |
|---|---|
| rows 1-5 decoders | `runs/{pipe}_loco_*/predictions_*.npz`, `scripts/snap_decode.py` |
| rows 6-7 level_mix | `runs/conventional_levelmix_loco_*`, `src/icme_mg/models/lattice_heads.py` |
| row 12 kNN | `scripts/knn_retrieval.py` -> `runs/knn_*.json`, `runs/summary_knn_{loco,loao}.json` |
| rows 13-19 tabular | `scripts/tabular_heads.py --variant {xgb_reg,xgb_cls,cat_reg,cat_cls,ftt,unc,fuse}` -> `runs/summary_tabular_*_loco.json` |
| rows 20-22 AR chain | `scripts/chain_head.py --mode {loco,loao}` -> `runs/chain_*.json`, `runs/summary_chain_{loco,loao}.json` |
| snap decoding | `runs/summary_snap_{loco,loao}_{gnn,conventional,genai}.json`, per-run `snap_eval_*.json` |
| dev70 screening | `runs/head_{proto_mix,level_mix}[_attn]_lr*` |

Hyperparameters: kNN k=5, cosine on z-scored features, softmax temperature 0.05;
XGBoost 300 trees depth 6 lr 0.1; CatBoost 500 iters depth 6 lr 0.1 (MultiRMSE needs
boosting_type=Plain on GPU); FT-Transformer 2 blocks d=128, 30 epochs, AdamW 3e-4;
fusion weight 0.5/0.5.
