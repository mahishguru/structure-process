# Story and Positioning: NeurIPS AI4Mat Workshop Paper

## Title (chosen)

**"Structure-to-Recipe: Calibrated Inverse Composition-Process Prediction for Mg Alloys"**

Alternates considered:
- "Closing the ICME Loop: Joint Probabilistic Inference of Composition and Process
  Parameters from Microstructure and Texture in Extruded Magnesium Alloys"
- "Which Alloy Made This Microstructure? Autoregressive Flow Inference of Composition and Process"

## One-paragraph pitch

ICME promises bidirectional traversal of the process-structure-property chain, yet
most ML work covers only the forward structure-to-property leg or, more recently,
property-to-structure generative inverse design. The remaining leg, inferring the
*recipe* (alloy composition and processing route) that produced an observed
microstructure, is the one a practicing metallurgist actually needs to act on a
designed microstructure. We pose it as conditional joint density estimation: given a
microstructure + texture descriptor, predict `p(composition, process | structure)`
with an autoregressive flow-transformer head. Because composition and process are
strongly interdependent (Gd content constrains feasible extrusion temperatures; Zn
alloys are extruded cold and fast), a factorized point regressor is wrong by
construction; our head captures the dependence via causal attention and returns
exact likelihoods, calibrated uncertainty, and diverse candidate recipes. We compare
three descriptor families feeding the identical head: (i) conventional
materials-informatics descriptors (GSH, Gram matrices, n-point statistics, grain
histograms), (ii) a pretrained generative vision latent (ViT-H/14 + FM-DiT bottleneck
from our inverse-design pipeline), and (iii) grain-graph GNN embeddings built from
DREAM.3D reconstructions and experimental micrographs.

## Why this fits AI4Mat

The AI4Mat CFP consistently asks for: (a) full-stack "real-world materials discovery"
pipelines rather than single-model benchmarks, (b) datasets/representations for
materials that go beyond crystals-and-molecules, (c) honest negative/ambiguity
results, and (d) uncertainty-aware methods bridging AI and experimental workflows.
We hit all four:

1. **Full-stack ICME loop.** We connect two published/submitted legs (Acta 2025
   forward S-P; NeurIPS 2026 inverse P-S) with the missing S-to-(C,P) leg, so the
   three papers jointly realize a closed design loop on one consistent alloy corpus.
2. **Representation study.** The workshop audience cares about "how should we
   represent microstructure for ML". Our three-pipeline comparison (hand-crafted
   statistics vs foundation-model latent vs relational grain graph) with an
   identical prediction head and identical folds is a clean, controlled answer.
3. **Ill-posedness treated honestly.** Different recipes can yield near-identical
   microstructures. Point regression hides this; a joint density exposes it as
   multimodality. Our candidate-shortlist figure (k-medoids over flow samples) shows
   the model recovering the *set* of plausible recipes.
4. **Calibration as headline metric.** With 14 as-extruded alloys and 108
   conditions, honest evaluation is leave-one-alloy-out extrapolation, where the
   right behavior is "know that you do not know". Quantile calibration and joint
   NLL are the differentiators, not MAE alone.

## Contributions (paper bullet list)

1. First joint-density inverse model `p(composition, process | microstructure)` for
   wrought Mg alloys: an autoregressive transformer with per-token neural spline
   flow heads (hurdle flows for sparse wt%, tolerance-based dequantization for
   quasi-discrete process labels).
2. A controlled comparison of three microstructure representation families
   (conventional statistics, generative foundation-model latents, grain-graph GNN)
   under an identical probabilistic head, identical splits, seeds, and budgets.
3. A curated benchmark: 17 extruded Mg alloy classes, 108 as-extruded
   alloy x temperature x speed conditions with composition labels, descriptors at
   four magnification scales, and frozen LOCO/LOAO splits (released).
4. Evidence that attention-based autoregression over labels beats independent heads
   (2x2 ablation: AR vs independent, flow vs Gaussian/MDN), because composition and
   process are physically coupled.

## Narrative arc (section flow)

1. **Intro.** ICME loop figure: property -> structure (done, cite NeurIPS sub),
   structure -> property (done, cite Acta), structure -> recipe (this work, red
   arrow). The metallurgist's question: "I have a target microstructure from
   inverse design. What do I melt and how do I extrude it?"
2. **Problem.** Formalize as conditional density estimation. Emphasize
   ill-posedness + label coupling. Why point regression and factorized heads fail.
3. **Method.** Descriptor pipelines (brief, cite prior work) -> adapters ->
   flow-transformer head. Physics-justified ordering (composition tokens before
   process tokens: alloy is chosen before the press is set). Tolerance
   dequantization as honest modeling of nominal-label uncertainty.
4. **Experiments.** LOCO (interpolation) and LOAO (extrapolation). Three pipelines
   x 5 heads/baselines. Headline: joint NLL + calibration + top-k alloy ID;
   secondary: per-label MAE/R2.
5. **Discussion.** Which representation carries recipe information? (Expectation
   from Acta SHAP: texture/GSH dominates alloy family identification, grain
   morphology carries process signal. Whether the GenAI latent, trained for
   reconstruction, retains process-relevant info is a genuinely open question and
   an interesting result either way.) Limits: nominal compositions, single
   extrusion route, 2D micrographs.

## Planned figures

| # | Figure | Purpose |
|---|--------|---------|
| 1 | ICME loop diagram with the three papers as arrows | positioning |
| 2 | Pipeline overview: 3 descriptor branches -> adapters -> shared AR flow head | method |
| 3 | Parity + per-label violin of flow samples vs truth (T, v, Gd, Zn) | accuracy |
| 4 | Reliability diagrams + PIT histograms, LOCO vs LOAO | calibration headline |
| 5 | Candidate-recipe shortlist: 5 k-medoid samples for one held-out micrograph, with joint log-likelihoods | ill-posedness story |
| 6 | Pipeline comparison bar chart (joint NLL, top-k alloy ID) with Wilcoxon markers | representation study |

Appendix: ordering ablation, 2x2 AR/flow ablation table, GNN graph construction
detail, dequantization sensitivity.

## Key risks and mitigations

- **Label memorization (only 108 unique label vectors).** LOAO is the honest test;
  frame LOCO as interpolation and LOAO as extrapolation stress test. Calibration
  metrics reward the model for knowing its uncertainty on unseen alloys.
- **GenAI checkpoint availability.** Latent extraction script is ready; checkpoint
  transfer from remote machine is a logistics task, not a research risk. Fallback:
  report conventional + GNN pipelines and add GenAI in camera-ready.
- **Descriptor leakage.** Acta-era PCA/Isomap reductions were fit on the full
  dataset. We refit all reducers inside each training fold.
- **HT conditions.** Excluded (13 of 121) because dropping the HT flag makes their
  labels collide with as-extruded twins. Revisit as a multimodality stress test if
  space permits.

## Tone and style

Concise, systems-flavored, honest about data scale. No em dashes. Emphasize that
this is the connective tissue between two larger works, purpose-built for a
workshop: complete loop, controlled comparison, released benchmark.
