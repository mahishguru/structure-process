# Architecture Specification: Autoregressive Flow-Transformer Head

Status: frozen v1 (2026-07). Implementation lives in `src/icme_mg/models/`.
Changes from the original design round: heat-treat flag dropped (user decision),
element set finalized from `data_labels.xlsx` as {Al, Zn, Mn, Ce, Nd, Y, Gd, Ca}
(trace Cu/Ni/Si/Fe are impurity level, < 0.012 wt%, excluded).

## 0. Overview

A small causal transformer decoder operating over a fixed-length sequence of
**10 label tokens** (8 elements + T_ext + v_ext), conditioned on the microstructure
descriptor via cross-attention over adapter tokens. Each continuous token's
conditional distribution is modeled with a conditional 1-D rational-quadratic
neural spline flow parameterized by the transformer hidden state. This yields the
exact tractable factorization

    p(y | c) = prod_{k=1..10} p(y_k | y_<k, c)

with exact per-token and joint NLL, ancestral sampling, and calibrated uncertainty.
The head is encoder-agnostic: each descriptor pipeline supplies conditioning through
a lightweight adapter into a shared interface. Head budget: ~1.6 M params.

## 1. Label tokenization

### 1.1 Sequence order (composition first, then process)

| Pos | Token | Type | Raw range (dataset) |
|-----|-------|------|---------------------|
| 1 | Al wt% | continuous, sparse | 0 - 2.88 |
| 2 | Zn wt% | continuous, sparse | 0 - 0.98 |
| 3 | Mn wt% | continuous, sparse | 0 - 2.1 |
| 4 | Ce wt% | continuous, sparse | 0 - 0.7 |
| 5 | Gd wt% | continuous, sparse | 0 - 10 |
| 6 | Ca wt% | continuous, sparse | 0 - 0.15 |
| 7 | Nd wt% | continuous, sparse | 0 - 0.57 |
| 8 | Y  wt% | continuous, sparse | 0 - 0.04 |
| 9 | T_ext (deg C) | continuous, few distinct values | 200 - 500 |
| 10 | v_ext (mm/s) | continuous, few distinct values | 0.5 - 7.5 |

Position 0 is a learned [BOS] token initialized from pooled adapter tokens.

Justification for ordering: (a) physics causality: composition is chosen before the
press is set, and the feasible (T, v) window is composition-constrained (high-Gd
alloys extrude at 350-450 C, never 200 C; Zn alloys at 250-400 C). Composition-first
lets the T and v flows condition on the full predicted composition through causal
attention. (b) Identifiability: composition is more strongly determined by the
descriptor (texture class separates alloy families) than process; resolve the more
identifiable block first. (c) Within composition, order by discriminative power:
Al, Zn, Mn separate AZ31/Z1/ME21 families, then RE elements. Report an ordering
ablation (process-first, one random permutation) in the appendix.

### 1.2 Sparse composition: hurdle (spike-and-slab) tokens

Most elements are zero for a given alloy. Each element token is a two-part hurdle:

    p(w_e | .) = (1 - pi_e) * delta_0(w_e) + pi_e * p_flow(w_e | .) * 1[w_e > 0]

pi_e = sigmoid(logit) from a linear head on the hidden state; p_flow is the spline
flow over the positive part. Exact-NLL compatible: zero labels contribute
-log(1 - pi_e); nonzero labels contribute -log pi_e - log p_flow(w_e). Flows cannot
place mass on an atom at 0, so this is required, and it gives element-presence
indicators for free (used for top-k alloy ID).

### 1.3 Normalization

- wt% (positive part): w~ = log(w + 0.1), then per-element z-score over nonzero
  training values (handles Ca 0.15 vs Gd 10 dynamic range).
- T: min-max to [-1, 1] over [200, 500]: T~ = (T - 350) / 150.
- v: log v, then z-score (grid 0.5 ... 7.5 is roughly log-uniform).
- Teacher-forced inputs: normalized scalar -> per-token linear lift R^1 -> R^d.
  Zero (absent) elements enter as a learned [ZERO] embedding, not the lifted value,
  so presence structure is explicit.

## 2. Transformer decoder

| Item | Choice |
|------|--------|
| Layers | 4 |
| Model dim d | 128 |
| Heads | 4 (head dim 32) |
| FFN | SwiGLU, hidden 256 |
| Norm | pre-RMSNorm |
| Masking | strict causal over 11 positions ([BOS] + 10 labels) |
| Dropout | 0.15 (attn, FFN, embedding) |
| Position/identity | learned label-identity embeddings (11 x 128); fixed schema, no sinusoidal PE |

### 2.1 Conditioning: cross-attention over adapter tokens

Each decoder layer: self-attn -> cross-attn -> FFN. Chosen over prefix tokens
(inflate the causal sequence, mix condition/label bookkeeping) and AdaLN (requires
pooling to one vector, discards the 16-token structure of the GenAI latent).

Adapters (only per-pipeline trainable difference; param counts reported in paper):

- **Conventional** (4-scale concatenated vector, D ~ 1.7-2.2k): Linear D -> 1024,
  reshape to 8 tokens x 128, + adapter position embeddings, + LayerNorm.
- **GenAI** (16 spatial tokens x 80): per-token Linear 80 -> 128, keep 16 tokens.
- **GNN**: multi-seed attention pooling of grain-graph node embeddings into
  8 super-node tokens x 128 (pooling is part of the GNN encoder).

### 2.2 Weight sharing across pipelines

Shared architecture, retrained per pipeline with identical hyperparameters, folds,
and seeds. A fully shared head would confound the representation comparison.
Optional appendix experiment: one universal head trained on all three.

## 3. Per-token flow head

Primary: conditional 1-D **rational-quadratic spline flow** (Durkan et al., NSF):
- 2 stacked spline layers (K = 8 bins, tail bound B = 4, linear tails) with one
  affine layer in between, all conditioned on h_k.
- Conditioner: shared trunk MLP 128 -> 128 (GELU, dropout 0.1) + per-token linear
  output heads emitting spline parameters (~50 outputs per layer).
- Base distribution N(0, 1).
- Element presence logits: per-element Linear 128 -> 1.

Rejected alternatives (kept as ablation/discussion):
- Per-token conditional flow matching: NLL needs ODE + trace estimator, poor
  cost/benefit for 1-D conditionals; mention as future work and tie-in to FM-DiT.
- MDN (per-token GMM, 5 components): ablation head; known variance-shrinkage and
  mode-collapse pathologies at this data scale; "flows beat MDN at equal budget".

## 4. Losses

    L = mean_k in cont [ -log p_flow(y_k | h_k) ]      (incl. hurdle terms, physical-unit Jacobians)
      + 1.0 * sum_e BCE(pi_e)                          (element presence)
      + 0.3 * CE_17(alloy class)                       (aux head on pooled adapter tokens; annealed to 0 over last 20% of training)

Label smoothing 0.05 on aux CE only, never on flow NLL. Reported joint NLL includes
the change-of-variables terms of all normalization maps, so it is in physical units
(nats per sample wrt wt%, deg C, mm/s).

## 5. Training

- Teacher forcing with **tolerance dequantization** (critical: T has ~8 distinct
  values, v ~11; flows degenerate on repeated atoms). Truncated-Gaussian jitter with
  physically justified scales: sigma_T = 5 C, sigma_v = 3% relative on log v,
  sigma_w = 2% relative on nonzero wt%. Fresh draw each epoch, identical draw
  applied to flow target and teacher-forced input. Same convention for all models
  when reporting NLL.
- Hierarchy: unit of generalization is the condition (alloy x T x v), not the
  image. All splits are condition-level; minibatches are per-condition balanced
  (uniform over conditions, then random instance).
- Splits: LOCO 5-fold over conditions (stratified so every alloy keeps >= 1 training
  condition) and LOAO 17-fold. Identical folds for all pipelines and baselines.
- Augmentation: conventional = descriptor jitter (sigma 0.02 on z-scored features);
  GenAI = native 81k augmented crops; GNN = node drop 5%, edge drop 5%, orientation
  jitter <= 2 deg. Conditioning dropout p = 0.1 (zero all adapter tokens).
- Optimizer: AdamW, lr 3e-4, betas (0.9, 0.95), weight decay 0.05 (none on norms,
  embeddings, flow-head biases), grad clip 1.0. Cosine decay, 5% warmup,
  ~200 balanced epochs, batch 64. EMA (0.999) for eval. Early stop on per-condition
  validation joint NLL, patience 25.

## 6. Inference

- Ancestral sampling, S = 256 joint samples per input (cheap: 10 tokens x 4 layers).
- Point prediction: per-label median of samples; element = 0 if pi_e < 0.5 else
  conditional median. Headline tables use the joint-MAP sample (highest joint
  log-density among S) so the (composition, process) tuple is self-consistent.
- Candidate shortlist: k-medoids (k = 5) over the S samples in normalized label
  space, reported with joint log-likelihoods (paper Figure 5).
- NLL: exact, one teacher-forced pass; aggregate per condition first.

## 7. Metrics

1. Per-label MAE (T in C, v in mm/s, wt% on nonzero truth), MAPE (T, v), R2,
   element-presence F1.
2. Joint NLL (nats/sample, physical units, fixed dequantization convention).
3. Quantile calibration: ECE over q in {0.05..0.95}, reliability diagrams, PIT.
4. Top-k alloy ID: nearest nominal composition (normalized L2) of joint-MAP sample,
   top-1/top-3; also from aux classifier for comparison.
5. Protocol: mean +/- std over folds; paired Wilcoxon on per-condition NLL for
   pipeline comparisons.

## 8. Baselines (2x2 attribution: autoregression x flows)

| Model | Isolates |
|-------|----------|
| XGBoost per-label (Acta recipe) | prior-work point-estimate anchor |
| GP per-label (RBF + White) | classical per-label uncertainty |
| Independent MDN (same adapters, MLP trunk, per-label GMM-5) | value of autoregression |
| AR transformer + Gaussian heads (identical decoder) | value of flows |
| AR transformer + MDN heads | flow vs mixture at matched budget |

## 9. Parameter budget

| Component | Params |
|-----------|--------|
| Adapter conventional (D -> 8x128) | 0.2 - 2.2 M (depends on D) |
| Adapter GenAI (16x80 -> 16x128) | 0.01 M |
| Adapter GNN (pool to 8x128) | 0.15 M |
| Embeddings ([BOS], [ZERO], identity, lifts) | 0.01 M |
| Decoder 4 x (self-attn + cross-attn + SwiGLU + norms) | 0.93 M |
| Flow conditioner trunk + per-token heads | 0.20 M |
| Presence / aux-class heads | 0.01 M |
| **Head total (excl. adapters)** | **~1.2 M** |

Headroom to 5 M allows d = 192 / 6 layers if underfitting (unlikely).

## 10. Literature anchors

- AR flows: Papamakarios et al. MAF (NeurIPS 2017); Durkan et al. Neural Spline
  Flows (NeurIPS 2019); Papamakarios et al. JMLR 2021 review.
- Transformer AR flows: Zhai et al. TarFlow (2024).
- Continuous-token AR heads: Li et al. MAR (NeurIPS 2024).
- Flow matching (discussion + FM-DiT tie-in): Lipman et al. (ICLR 2023); Liu et al.
  Rectified Flow (ICLR 2023); Tong et al. CFM (TMLR 2024).
- MDN: Bishop 1994. Dequantization: Theis et al. 2016; Ho et al. Flow++ 2019.
- Conditioning: Peebles & Xie DiT (ICCV 2023), for the AdaLN-vs-cross-attn argument.
- Materials: Acta Materialia 2025 (descriptors, XGBoost/GP recipe); NeurIPS 2026
  submission (encoder source); Kalidindi hierarchical materials informatics;
  Paulson et al. Acta 2017; inverse PSP works in AM; generative inverse design
  surveys.
