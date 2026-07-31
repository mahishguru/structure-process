# Figure Plan: Structure-to-Recipe (AI4Mat)

Every figure earns its place by advancing one story beat: forward ICME is solved,
the inverse recipe leg is not, we solve it with calibrated joint densities, and we
answer which representation carries the recipe signal. Workshop papers get 4 pages;
target 5 main figures + 1 main table, push the rest to the appendix.

## Global conventions (apply to every figure)

- Output: vector PDF (`fig/figN_name.pdf`), fallback 300 dpi PNG for drafts.
- Sizes: single column = 5.5 in wide (NeurIPS `\textwidth`), half figures 2.65 in.
  Height at most 0.6x width unless a grid demands more.
- Fonts: match the paper body. `matplotlib` rcParams:
  `font.family: serif`, `font.size: 8`, `axes.labelsize: 8`, `legend.fontsize: 7`,
  `xtick/ytick.labelsize: 7`. No titles inside the axes; captions carry the prose.
- Colors: one fixed colorblind-safe mapping used in EVERY figure so pipelines are
  instantly recognizable:
  - Conventional = `#0173B2` (blue)
  - GenAI = `#DE8F05` (orange)
  - GNN = `#029E73` (green)
  - Reference heads (XGBoost/GP/MDN) = greys `#949494`, `#5A5A5A`
  - Truth / reference lines = black, dashed
- LOCO vs LOAO: LOCO = filled markers/solid lines, LOAO = open markers/dashed.
- Seeds/folds: always plot the fold-mean with per-fold points (small, alpha 0.4),
  never bare means. Error bars = 95% CI over folds.
- Scripts live in `scripts/figures/` (one script per figure,
  `python scripts/figures/fig3_parity.py --runs runs/ --out paper/fig/`),
  deterministic given the run directory, no hand editing.
- Every script ends by printing the exact numbers used, so captions can quote them.

---

## Figure 1 - The ICME loop and the missing leg (positioning, hand-drawn schematic)

**Story beat.** Property-to-structure (NeurIPS sub) and structure-to-property
(Acta 2025) exist on this exact corpus; structure-to-recipe is the missing leg
that turns a designed microstructure into an actionable melt-and-extrude order.

**Content.**
- Triangle (or ring) of three nodes: Property, Structure, Recipe (composition +
  process). Three arrows:
  - Structure -> Property: grey, solid, labeled "forward surrogate (Acta 2025)".
  - Property -> Structure: grey, solid, labeled "inverse design (NeurIPS sub)".
  - Structure -> Recipe: RED, bold, labeled "this work: p(composition, process | structure)".
- Under the Structure node: one real micrograph thumbnail (pick a visually clean
  150 um AZ31 codec render from `data/genai/images/`).
- Under the Recipe node: a tiny table stub "8 wt% + T + v" to make the target concrete.
- Optional inset question: "which alloy made this, and how was it extruded?"

**Data/tooling.** TikZ inside the paper (preferred: crisp at any scale) or
Inkscape SVG -> PDF. No script needed; keep source in `paper/fig/fig1.tex`.

**Pitfalls.** Do not let this become a generic ICME cartoon; the three arrows must
name the three papers so the "closing the loop" claim is visual, not rhetorical.

---

## Figure 2 - Method overview: three representations, one head (method)

**Story beat.** The controlled comparison IS the contribution: three descriptor
families feed the *identical* autoregressive flow-transformer head under identical
splits, seeds, and budgets.

**Content.** Three horizontal branches converging into one head, left to right:
1. Conventional branch (blue): micrograph -> icons for 3-pt correlations, Gram,
   AR/grain-size histograms, GSH texture -> "438-D vector" -> VectorAdapter
   (-> 8 x 128 tokens).
2. GenAI branch (orange): micrograph -> RVE codec render -> frozen ViT-FMDiT-1280
   encoder (snowflake icon = frozen) -> 16 x 80 latents -> TokenAdapter
   (-> 16 x 128 tokens).
3. GNN branch (green): micrograph -> DREAM.3D RVE -> grain-adjacency graph
   (nodes = grains w/ orientation quaternion, edges = shared boundary +
   misorientation) -> 4-layer GATv2 -> attention pooling (-> 8 x 128 tokens).
   Mark this branch "trained end-to-end" (flame icon) to contrast the frozen GenAI.
4. Shared head (right): causal transformer over 10 label tokens (8 elements then
   T, v; annotate "alloy chosen before the press is set"), cross-attention to the
   descriptor tokens, per-token output = hurdle spike-and-slab + RQ-spline flow.
   Show one sampled recipe emerging: "Gd 9.8 wt%, Mn 0.4, ..., T 400 C, v 2 mm/s".

**Data/tooling.** TikZ or draw.io/Inkscape -> PDF. Real thumbnails: one micrograph,
its codec render, and a real graph rendering (see Fig A4 script) instead of icons
where feasible - reviewers reward real artifacts.

**Pitfalls.** Keep adapter/token dimensions in the figure (8x128 / 16x128); the
"identical head" claim needs the visual proof that all three branches emit the
same interface. Do not draw the DiT decoder; we only use the encoder.

---

## Figure 3 - Can it recover the recipe? Parity + posterior spread (accuracy)

**Story beat.** Before selling calibration, show the model is *accurate* where
accuracy is possible: process parameters and major alloying elements on LOCO.

**Content.** 2 x 2 grid (single column), each panel one label:
T (extrusion temperature), v (ram speed), Gd, Zn.
- X = true value, Y = flow posterior over S=256 ancestral samples per test
  condition: plot the posterior median as a marker and a thin vertical bar for the
  16-84% quantile range. Identity line dashed black.
- Color by pipeline; to avoid overplotting show the best pipeline per panel and
  put the full 3-pipeline versions in the appendix (Fig A2). Alternatively small
  multiples per pipeline if space allows.
- Aggregate per condition (108 points max per panel), not per image: predictions
  for the ~40 images of one condition are averaged in sample space first.
- Annotate each panel with MAE and R2 (fold-mean).

**Data.** `runs/{pipeline}_flow/loco_fold*/eval_test.json` (per-sample posterior
samples or quantiles; if eval json stores only metrics, add a `--dump-samples`
eval flag first). Truth from `data/labels/labels.csv`.

**Script.** `scripts/figures/fig3_parity.py`.

**Pitfalls.**
- Zero-inflated elements: for Zn/Gd only plot conditions where the element is
  present; the hurdle presence accuracy goes in the text/Table 1, not here.
- Ram speed is quasi-discrete (nominal levels); jitter x slightly and say so in
  the caption, or the identity line looks artificially banded.

---

## Figure 4 - Calibration under interpolation vs extrapolation (headline)

**Story beat.** The differentiator of this paper: on unseen alloys (LOAO) the
model should *know that it does not know*. Calibration, not MAE, is the metric
that shows it.

**Content.** Two panels side by side (full width):
- Left: reliability diagram (empirical coverage vs nominal central-interval
  level, 10-90%) for the joint continuous labels. One line per pipeline (3
  colors), LOCO solid / LOAO dashed = 6 curves. Diagonal = ideal. Shade the
  miscalibration area of the best pipeline lightly.
- Right: PIT histograms for T and v (the two labels a metallurgist acts on),
  best pipeline, LOCO vs LOAO overlaid (LOAO as step outline). Uniform = ideal,
  drawn as dashed horizontal line.
- Caption quotes ECE numbers for all six curves.

**Data.** Same eval dumps as Fig 3; PIT values need per-sample CDF evaluations -
the flow head gives exact CDFs via the spline transform, store them at eval time
(`eval_*.json: pit[label][sample]`).

**Script.** `scripts/figures/fig4_calibration.py`.

**Pitfalls.**
- Compute PIT only over present (non-zero) elements for hurdle labels.
- Per-condition aggregation again: one PIT value per condition per label, else
  image-replicates inflate the sample count and flatten the histograms.
- If LOAO turns out badly calibrated everywhere, that IS the honest result -
  show it and discuss; do not tune it away silently.

---

## Figure 5 - Ill-posedness made visible: candidate recipe shortlist (story)

**Story beat.** Different recipes can produce near-identical microstructures.
A point regressor hides this; our joint density exposes it as multimodality and
returns a *shortlist* a metallurgist can act on.

**Content.** One held-out micrograph (choose an ambiguous LOAO case, e.g. a
Mg-Gd condition whose texture resembles a neighboring Gd level):
- Left: the micrograph (codec render) + its true recipe as a small header row.
- Right: 5 k-medoid representatives of the 256 flow samples, drawn as horizontal
  recipe cards: 10 mini-bars (8 elements wt% + T + v) with the true recipe
  overlaid as black ticks. Each card annotated with its joint log-likelihood and
  cluster weight (% of samples in that medoid's cluster).
- Highlight: one medoid is (close to) the true recipe; another is a plausible
  alternative - caption explains why both are physically viable.
- Optional bottom strip: 2-D marginal density (Gd vs T) contour from the samples
  showing two modes, true recipe as star.

**Data.** Flow samples for the chosen condition from the eval dump; k-medoids
(k=5, Gower or standardized L2 in label space) computed in the script.

**Script.** `scripts/figures/fig5_shortlist.py --condition <cid>`.
Selection procedure for the showcase condition: rank LOAO test conditions by
posterior multimodality (e.g. silhouette of k=2 clustering of samples) and pick
the top one whose modes are chemically interpretable; print the ranking so the
choice is reproducible, not cherry-picked silently.

**Pitfalls.** This figure sells the paper - iterate on it early with dummy
samples (the script should accept any samples array) so layout is settled before
real runs land.

---

## Figure 6 - Which representation carries the recipe? (representation study)

**Story beat.** The AI4Mat question: hand-crafted statistics vs foundation-model
latent vs relational grain graph - which retains provenance information, and does
the frozen generative latent (trained for reconstruction) keep process signal?

**Content.** Two panels (full width, or stacked single column):
- Left: grouped bars, x = pipeline (3 groups), y = joint NLL (physical units,
  lower better), LOCO and LOAO bars side by side per pipeline (solid/hatched),
  per-fold points overlaid, 95% CI whiskers. Wilcoxon significance brackets
  between pipeline pairs (report p in caption).
- Right: top-1 / top-3 alloy identification accuracy (from posterior samples,
  nearest-alloy assignment), same grouping. Include grey reference bars for the
  XGBoost and MDN reference heads (conventional features) so "the head matters too" is
  visible in the same glance.

**Data.** `runs/*/eval_{loco,loao}.json` metric fields; Wilcoxon over the
19 folds pairing.

**Script.** `scripts/figures/fig6_pipelines.py`.

**Pitfalls.** Joint NLL scales differ LOCO vs LOAO; if LOAO dwarfs LOCO use two
y-scales or split panels rather than compressing LOCO differences to invisibility.

---

## Main table (not a figure, but plan it with the figures)

Table 1 = the 2x2 head attribution + pipeline sweep already stubbed in main.tex:
rows = pipeline x head (XGBoost, indep. MDN, AR+Gaussian, AR+MDN, Flow-AR),
columns = joint NLL, MAE(T), MAE(v), ECE, presence F1, top-1 alloy. LOCO main,
LOAO in appendix twin. Populated by `scripts/figures/table1_main.py` emitting
LaTeX rows directly from `runs/*/eval_*.json`.

---

## Appendix figures

**A1 - Dataset atlas.** 4 x 4 grid: four alloys (rows) x four magnifications
(columns) of codec renders, labeled with condition ids; plus a half-width panel
with the label-space scatter (T vs v, marker = alloy family). Establishes corpus
breadth and the quasi-discrete label grid. Script: `figA1_atlas.py` from
`data/gnn/rves/manifest_*.csv` + `labels.csv`.

**A2 - Full parity grids.** Fig 3 repeated for all 10 labels x 3 pipelines,
LOCO and LOAO. Script: reuse `fig3_parity.py --all`.

**A3 - 2x2 ablation visual.** Joint NLL bars for {independent, AR} x {Gaussian/MDN,
spline flow} on conventional features; shows AR and flexible marginals are both
needed. Script: `figA3_ablation.py`.

**A4 - Graph construction detail.** One RVE: codec render -> FeatureIds
segmentation -> overlaid graph (nodes at centroids sized by log-area, edges
colored by misorientation angle 0-90 deg colorbar). Pick a mid-density 120 um RVE
(~200 grains) for legibility. Script: `figA4_graph.py --stem <rve_stem>` reading
the `.dream3d` + `.pt`.

**A5 - Latent structure sanity check.** UMAP (or PCA) of the 4348 GenAI latents
(mean-pooled over 16 tokens), colored (a) by alloy family, (b) by extrusion T.
Answers "does the frozen reconstruction latent organize by recipe at all?"
independently of the head. Script: `figA5_latent_umap.py` from
`data/genai/latents.npz`.

**A6 - Per-scale contribution.** Joint NLL when training on single magnification
subsets (90/100/120/150) vs all four, per pipeline. Justifies the multi-scale
corpus. Script: `figA6_scales.py` (requires the per-scale training runs).

**A7 - Calibration extras.** Reliability diagrams per label and PIT histograms for
all labels/pipelines; dequantization-tolerance sensitivity sweep for T and v.

---

## Build order and dependencies

| When | Figures | Blocked by |
|---|---|---|
| Now (no runs needed) | 1, 2, A1, A4 | nothing - data on disk |
| After latents land | A5 | `data/genai/latents.npz` |
| Layout now, numbers later | 3, 4, 5, 6, Table 1 | training + eval dumps |
| After ablation runs | A3, A6, A7 | ablation configs |

Recommended immediate actions: draft Figs 1-2 (TikZ) and write `fig5_shortlist.py`
against synthetic dummy samples so the flagship figure's layout is converged
before any model finishes training. Ensure the eval loop dumps per-sample
posterior draws, PIT values, and per-condition ids - every downstream figure
depends on that dump format.
