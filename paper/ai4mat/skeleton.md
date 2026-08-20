# Skeleton: AI4Mat-NeurIPS-2026 paper

Conforms to manifesto.md. Full-length, 9 pages + references + appendix.

## Title

**"Which Alloy Made This, and How Was It Extruded? Structure-to-Recipe
Inference on Sparse Experimental Data"**

The question is the exact caption already on the loop figure. The subtitle
carries the method and the data regime.

## Abstract (draft sketch)

Three moves: (1) real experimental Mg campaign data is sparse and messy, 107
conditions x 14 alloys, and the inverse provenance question is the missing leg
of the loop; (2) we benchmark three descriptor families against task-aligned
heads under honest condition-grouped 5-fold CV with guard metrics; (3) answer:
handcrafted statistics + a lattice-aware reranker win composition (13.6%
WAPE), a catalogue-decoding GP wins the process window, and scaling the
representation does not help at this data scale.

## Section plan and page budget

| # | section | pages | content |
|---|---|---:|---|
| 1 | Introduction | 1.25 | Fig 1 (missing leg). The question. Why provenance matters for automated characterization and QC in self-driving labs. Contributions list (4 bullets). |
| 2 | Data: sparse by construction | 1.0 | 107 conditions, 14 alloys, 8-element lattice, OM + XRD texture. Sparsity along both axes; family-confounded process sub-grids. Two symmetric tasks A/B with known-other-half conditioning. Leakage audit: image-random split memorizes (100% top-1), condition-grouped 5-fold CV is the protocol. |
| 3 | Three ways to describe a microstructure | 1.75 | Fig 2 (pipeline). Conventional 438-D statistics. Vision embedding 1280-D: ViT-H trunk + attention pooler to 16x80 spatial latents, frozen MMDiT decoder arm trained jointly under the 5-term composite loss (Eq. 1: flow-matching velocity + clean-latent rec + InfoNCE + VICReg + FFT) (Fig vision). GNN 128-D grain-graph embedding with the four-stage figure inline (Fig 3). |
| 4 | Heads that match the label space | 1.25 | Composition = lattice classification (kNN, balanced fusion, OOF-constrained reranker); process = catalogue decoding (trees, ARD GP, joint-grid GP + ordinal velocity). Metrics with guard columns. Reference placeholders on all method heads. |
| 5 | Results | 2.5 | Table 1 (composition) + Table 2 (process) + per-fold tables (Task A per-fold, Task B full stats) inline. Findings 1-4, guard-metric lesson, calibration honesty. |
| 6 | What this means for messy real data | 0.75 | Scaling answer, descriptor compatibility, what to measure first, limitations. |
| 7 | Conclusion | 0.25 | Restate the question and the answer. |
| 8 | Reproducibility | 0.25 | run_cv.py / aggregate_cv.py, raw predictions stored. |

## Figure map

| fig | file | where | caption job |
|---|---|---|---|
| 1 | figures/fig1_missing_leg.png | S1, top | The loop with the missing leg; pose the question visually. |
| 2 | figures/fig2_pipeline.png | S3 | Three descriptor branches into shared heads; lattice classification + GP regression. |
| 3 | figures/fig_vision_embedding.png | S3 (vision embedding paragraph) | The vision-embedding branch: ViT-H encoder, MMDiT decoder, 16x80 latents to 1280-D. |
| 4 | figures/fig3_gnn.png | S3 (grain-graph paragraph, inline) | GNN four-stage detail: graph, GATv2 x4, attention pooling, 8 tokens x 128. |

## Table map

| tab | source | content |
|---|---|---|
| 1 | composition_heads_cv.csv (mean_std) | 3 heads x 3 representations: WAPE +/- std, top-1, plus all-WAPE + FP; winner bold |
| 2 | process_heads_cv.csv (mean_std) | 3 heads x 3 representations x 2 targets: MAE, MAPE, R2; winner bold |
| 3 | composition_heads_cv.csv (fold rows) | Task A per-fold WAPE / top-1 (in Results) |
| 4 | process_heads_cv.csv (mean_std) | Task B full mean +/- std (in Results) |
| (supp) | cv_selection_audit.csv | selected owner weights and velocity decoders per fold |

## Numbers lockbox (only these appear in the paper)

Composition (WAPE % / top-1):
- conv: reranker 13.6+/-4.1 / 0.646+/-0.051; fusion 16.8+/-5.8 / 0.612+/-0.078; kNN 30.0+/-7.6 / 0.496+/-0.110
- vision: reranker 41.6+/-18.9 / 0.287; fusion 44.7+/-18.0 / 0.276; kNN 57.2+/-16.4 / 0.294
- gnn: reranker 44.2+/-8.0 / 0.266; fusion 44.3+/-8.0 / 0.266; kNN 44.8+/-11.7 / 0.279
- conv reranker guards: all-WAPE 19.7%, FP 1.1%, NLL 1.38
- fusion guards: all-WAPE 25.4%, FP 1.3%; kNN: 61.7%, FP 10.3%

Process (MAE / MAPE % / R2):
- T_ext: trees 41.6/11.9/0.349 (conv), 39.2/11.4/0.391 (vision), 36.5/10.7/0.443 (gnn); ARD GP 52.1/15.2/0.039 (conv), 72.1/19.9/-0.813 (vision), 49.5/14.7/0.084 (gnn); grid GP 38.1/10.9/0.326 (conv), 35.2/10.5/0.367 (vision), 36.6/10.8/0.357 (gnn)
- v_ext: trees 1.34/64.5/0.281 (conv), 1.22/55.9/0.409 (vision), 1.11/56.0/0.525 (gnn); ARD GP 1.71/104.7/-0.240 (conv), 1.76/120.5/-0.200 (vision), 1.44/94.5/0.012 (gnn); grid GP 1.05/49.9/0.535 (conv), 1.12/50.0/0.447 (vision), 1.12/53.3/0.473 (gnn)
- MAPE floors (constant predictor): T_ext 14.5%, v_ext 136.5%
- coverage90: grid GP conv T 0.52 / v 0.56; gnn ARD GP T 0.69 / v 0.71

Dataset: 107 conditions, 14 alloys, 8 elements; T_ext 200-500 C, v_ext
0.5-7.5 mm/s; 15-108 images per condition; ~4,300 images total.

## Writing rules for main.tex

- Every paragraph earns its place; no throat-clearing.
- No em dashes. Short sentences where possible.
- Red \needref{...} at every citation point with a precise hint.
- Self-refs: Guru2025 (Acta) and CoPiLOT2026 (under review) as real entries.
- Bold only the winning numbers in tables.
