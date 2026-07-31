# Final Paper Results (source of truth)

Story: inverse structure->recipe on scarce data. Three findings:
(1) representation comparison under a shared probabilistic head (conventional
descriptors carry the most provenance information); (2) continuous density
estimation is the wrong output type for a discrete composition lattice; typed
discrete prediction (classification + retrieval fusion) halves the error; (3)
random image splitting is fully leaked because repeated images of every test
condition occur in training, so the study uses stratified random folds over
whole conditions.

The final fusion gives each processing condition equal total training weight
and removes a validation-selected power of the empirical alloy-frequency
prior. This matches the condition-level estimand despite the observed 15-108
images per condition and reduces bias against alloys represented by only four
conditions.

Flow-AR is kept in the paper in two roles only: the shared head for the
representation comparison (Table 1) and the continuous-density baseline in the
head study (Table 2). It is not the proposed method.

## Table 1: Representation comparison (random condition-grouped 5-fold CV, shared flow-AR head)

| pipeline | joint NLL | presence F1 | top-1 | top-3 | ECE | element MAE |
|---|---|---|---|---|---|---|
| Conventional | 2.460 +/- 2.826 | **0.648 +/- 0.132** | **0.31** | **0.46** | **0.299** | **0.478** |
| GenAI latents | **-1.150 +/- 1.744** | 0.553 +/- 0.073 | 0.20 | 0.34 | 0.322 | 0.550 |
| GNN (90 um only) | 0.553 +/- 0.479 | 0.394 +/- 0.162 | 0.16 | 0.28 | 0.320 | 0.709 |

Message: generative latents win density modeling; conventional descriptors win
every decision metric; all flows are overconfident (cov90 0.18-0.37 vs 0.90).

## Table 1b: Representation comparison under the SAME winning head
(fusion = CatBoost cls + kNN, identical code path for all three; GNN uses
embeddings extracted from the trained per-fold GATv2 encoders, mean-pooled
tokens; genai uses flattened 16x80 latents. Random condition-grouped 5-fold CV.
source: runs/summary_unified_{conventional,genai,gnn}_loco.json)

| pipeline | el. WAPE (fuse) | el. MAE | top-1 | top-3 | class NLL | T WAPE |
|---|---|---|---|---|---|---|
| Conventional | **21.7%** | **0.292** | **0.54** | **0.68** | **1.66** | 13.1% |
| GenAI latents | 57.5% | 0.485 | 0.29 | 0.50 | 2.44 | 16.1% |
| GNN (90 um) | 57.2% | 0.503 | 0.27 | 0.44 | 2.69 | 14.1% |

Message: the representation ranking is HEAD-INVARIANT. Conventional wins under
both the flow head and the discrete fusion head; the head-study conclusion is
not an artifact of the conventional pipeline. (Per-head detail: kNN/cat/fuse
all available per pipeline in the summary files.)

## Table 2: Head study (conventional features, random condition-grouped CV) - the core table

| head | element WAPE | element MAE | top-1 | top-3 | T_ext WAPE |
|---|---|---|---|---|---|
| Flow-AR (MAP decode) | 44.8% | 0.478 | 0.31 | 0.46 | 14.8% |
| Flow-AR + lattice-snap decode | 45.1% | 0.412 | 0.30 | 0.49 | 14.0% |
| kNN retrieval (k=5) | 34.3% | 0.374 | 0.48 | 0.61 | 13.1% |
| XGBoost 14-way classifier | 36.6% | 0.346 | 0.40 | 0.66 | 13.1% |
| CatBoost 14-way classifier | 25.7% | 0.322 | 0.50 | 0.68 | 13.1% |
| FT-Transformer | 30.2% | 0.396 | 0.46 | 0.62 | **12.1%** |
| AR level chain (alloy-scored) | 35.4% | 0.325 | 0.45 | 0.62 | 15.0% |
| CatBoost + kNN fusion | 21.7% | 0.292 | 0.54 | **0.68** | 13.1% |
| Condition-balanced CatBoost + kNN fusion | 21.4% | 0.244 | **0.58** | **0.68** | 12.8% |
| **Prior-adjusted condition-balanced fusion** | **19.1%** | **0.221** | **0.58** | **0.68** | 12.8% |
| AR level chain + kNN fusion | 26.0% | **0.273** | 0.51 | 0.66 | 15.0% |

Message: composition on a 14-alloy lattice is a classification problem;
fusing a condition-balanced GBDT classifier with condition-balanced retrieval
and validation-selected long-tail correction reduces the flow's element WAPE
by 57% while nearly doubling alloy identification.

## Split-unit diagnostic

Random image-stratified 5-fold CV with the same CatBoost + kNN fusion gives
100% alloy top-1/top-3, zero composition WAPE and MAE, and 0.12% temperature
WAPE. The train/test condition-overlap fraction is 1.0 in every fold. These are
memorization results, not generalization results, and are excluded from the
headline tables. The primary protocol randomly assigns whole conditions to
alloy-stratified folds.

## Calibration and uncertainty (text/small table)

- Composition UQ = the fused 14-way distribution over nominal compositions:
  class NLL 1.66 (vs 1.95 CatBoost alone, 5.08 kNN alone);
  mean |confidence - accuracy| 0.32 -> temperature scaling / conformal sets.
- Process UQ WINNER: exact GP regression (RBF+White kernel, condition-level,
  PCA-32 features) is essentially calibrated out of the box:
  T_ext 90% coverage 0.858 conventional / 0.819 genai / 0.840 gnn
  (target 0.90), T WAPE 13.3-16.4%. NGBoost coverage 0.44, CatBoost
  RMSEWithUncertainty 0.07: both severely overconfident. GP is the process
  head. n~86 training conditions is exactly the GP regime.
- GP classifier over the alloy lattice is weak (top-1 0.17-0.31): condition
  pooling discards the multi-image evidence that boosting exploits; GBDT
  fusion stays the composition head.
- GMM/MDN + AR heads are the flow-AR family already benchmarked (Table 2):
  continuous mixtures share the flow's failure mode (mass off-lattice), so
  the lattice argument covers them.
- Flow-AR calibration: ECE ~0.30, cov90 0.18-0.37 in all pipelines
  (motivates the discrete UQ pivot).

## Cut from the paper (kept in results/archive/ for reference)

- proto_mix / level_mix lattice heads and dev70 screening (level_mix appears
  indirectly: lattice-snap decode row retains the idea).
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
| Table 1 | runs/summary_loco.json (gnn), runs/summary_loco_{conventional,genai}.json |
| Table 2 | runs/summary_tabular_*_loco.json, including summary_tabular_fuse_balanced_loco.json; runs/summary_logit_adjusted_fusion.json; runs/summary_knn_loco.json; runs/summary_snap_loco_conventional.json; runs/summary_chain_loco.json |
| Table 1b + GP UQ | runs/summary_unified_{conventional,genai,gnn}_loco.json |
| split diagnostic | runs/summary_random_image_fuse.json |
| scripts | scripts/{tabular_heads,logit_adjusted_fusion,knn_retrieval,chain_head,snap_decode,unified_heads}.py |
| gnn embeddings | data/gnn/embeddings/loco_fold*.npz (cached from trained encoders) |
| full detail | results/archive/{ablation_tables,prediction_head_ablation}.md |
