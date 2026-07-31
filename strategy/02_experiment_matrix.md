# Experiment Matrix

Every cell below runs with identical frozen splits (data/splits/), identical seeds
{0, 1, 2}, and the dequantization convention of strategy/01 section 5.

## Axes

- **Pipelines (conditioning):** conventional | genai | gnn-synthetic | gnn-experimental
  (gnn-synthetic = DREAM.3D RVE graphs, gnn-experimental = binarized OM/EBSD graphs;
  main text reports the better of the two as "gnn", the other goes to appendix)
- **Heads:** flow-AR (primary) | gaussian-AR | mdn-AR | mdn-independent
- **Classical baselines (per-label):** xgboost | gp
- **Splits:** LOCO (5-fold over 108 conditions) | LOAO (14-fold over base alloys)

## Main-text table (Table 1)

| Row | Pipeline | Head | Splits |
|-----|----------|------|--------|
| 1 | conventional | xgboost | LOCO + LOAO |
| 2 | conventional | gp | LOCO + LOAO |
| 3 | conventional | mdn-independent | LOCO + LOAO |
| 4 | conventional | gaussian-AR | LOCO + LOAO |
| 5 | conventional | flow-AR | LOCO + LOAO |
| 6 | genai | flow-AR | LOCO + LOAO |
| 7 | gnn | flow-AR | LOCO + LOAO |

Rows 1-5 form the head ablation on the conventional pipeline (cheapest). Rows 5-7
form the representation comparison at fixed head.

Columns: joint NLL, calibration ECE, top-1/top-3 alloy ID, MAE(T), MAE(v),
MAE(Gd), MAE(Zn), presence F1. Mean +/- std over folds x seeds.

## Appendix tables

- A1: full per-label MAE/MAPE/R2 for every row above.
- A2: mdn-AR vs flow-AR (flow vs mixture at matched budget).
- A3: token-ordering ablation (composition-first vs process-first vs random perm),
  conventional + flow-AR, LOCO only.
- A4: gnn-synthetic vs gnn-experimental.
- A5: dequantization sensitivity (sigma_T in {2, 5, 10} C).
- A6: single-scale (150 um) vs 4-scale-concat conventional descriptors.
- A7: universal head (one head trained across all pipelines) vs per-pipeline.

## Compute plan

Head training is trivial (~1.6 M params, 108 conditions of unique labels,
instance-balanced batches). One LOCO sweep = 5 folds x 3 seeds ~ minutes/run on a
single GPU or CPU. The expensive parts are one-time dataset builds:
- conventional: descriptor assembly + per-fold PCA/Isomap refits (CPU, hours)
- genai: latent extraction over image corpus (single GPU pass; needs checkpoint)
- gnn: DREAM.3D RVE generation per condition + graph building (CPU, hours), GNN
  encoder pretraining or joint training with head (single GPU, minutes-hours)

## Decision gates

1. After labels + conventional pipeline: run rows 1-5 LOCO. Gate: flow-AR beats
   mdn-independent on joint NLL and matches XGBoost on MAE(T). If not, debug
   dequantization and balance before scaling out.
2. After gate 1: LOAO sweep + calibration figures. Expect degradation; the story
   is calibrated degradation.
3. GenAI + GNN rows added independently as their datasets land.
