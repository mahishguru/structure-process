# AI4Mat-NeurIPS-2026 manifesto

Fixed decisions and constraints for the workshop paper. Everything in
skeleton.md and main.tex must conform to this file.

## Venue facts

- AI4Mat (AI for Accelerated Materials Design) workshop, NeurIPS 2026, Sydney,
  December 2026. Submission via OpenReview, double-blind.
- Two length classes: short-form up to 4 pages (work in progress) or
  full-length up to 9 pages (polished, complete), both with unlimited
  references and supplementary. 6-page in-between submissions are explicitly
  discouraged.
- Non-archival; accepted papers get a poster, some a spotlight.

## Track decision: Paper Track, not the Translational AI themed track

The translational track requires "experimental or field validation and/or
evidence of use with an external or industrial partner". This work has neither:
no deployment, no industrial partner, no closed-loop lab integration.

The Paper Track fits exactly under its Automated Characterization pillar:
"analysis of real-world characterization data, e.g. microscopy data (including
multi-modal data like images, spectra and diffraction patterns)" and "machine
learning algorithms for small-data regimes". The paper analyzes real optical
micrographs and XRD texture measurements and infers the recipe that produced
them.

Length: full-length, 9 pages. The work is complete (finished benchmark,
finished results), so short-form would undersell it and full-length reviewer
standards are fine because the evaluation is honest and reproducible.

## Session alignment

Frame toward the session "Automating Discovery That Delivers: When AI Meets
the Messiness of Real Experiments". Our messiness is concrete:

- 107 measured conditions across 14 alloys, pooled from published extrusion
  campaigns and in-house experiments; not a curated simulation benchmark.
- Sparse along both task axes: the composition lattice (14 alloys on an
  8-element simplex) and the process grid (T_ext 200-500 C, v_ext 0.5-7.5
  mm/s) are both thinly sampled, and the two axes are confounded (each alloy
  family lives on its own process sub-grid).
- Repeated images per condition make naive random splits fully leaky; honest
  evaluation needs condition-grouped 5-fold cross-validation.
- A headline metric (present-element WAPE) that is silently gameable unless
  guard metrics travel with it.

Also touch the "Scaling Laws for Materials Reasoning" session question with a
direct empirical answer for this task: foundation-model latents (1280-D,
2.5B-parameter lineage) and learned graph embeddings do not beat 438-D
handcrafted statistics; aligning the prediction head with the discrete label
structure matters more than scaling the representation.

## Story spine

1. The ICME loop has a missing leg: forward structure-to-property exists
   (Acta 2025), inverse design of structure from property targets exists
   (NeurIPS 2026, under review), but structure-to-recipe provenance
   (which composition, which extrusion window) is open.
2. Pose it as the title question: which alloy made this, and how was it
   extruded?
3. Decompose into two symmetric tasks with known-other-half conditioning:
   Task A composition given process, Task B process given composition.
4. The label spaces are discrete: a 14-alloy lattice and a catalogue of
   observed (T, v) pairs. Heads that respect this structure win; continuous
   regression loses.
5. Descriptor compatibility is the representation finding: handcrafted
   microstructure/texture statistics are the most compatible with recipe
   inference at this data scale; the generative latent is the least.
6. Evaluation honesty is a contribution, not a footnote: condition-grouped
   5-fold CV, all-element WAPE and false-positive guards next to the headline
   WAPE, selection on training out-of-fold predictions only.

## Claims we are allowed to make (from results/tables/*_cv.csv, 5-fold)

- Composition, conventional + OOF-constrained reranker: WAPE 13.6 +/- 4.1%,
  all-WAPE 19.7%, FP 1.1%, top-1 0.646 +/- 0.051. Beats the balanced-fusion
  comparator (16.8 +/- 5.8%, top-1 0.612) on every metric.
- The reranker only helps on conventional: on genai/gnn the OOF selection
  picks owner weight 1.0 on 8 of 10 folds, i.e. the learned-representation
  auxiliaries carry no complementary signal.
- Process, joint-grid GP: v_ext MAPE 49.9 / 50.0 / 53.3% (conv/genai/gnn),
  best on all three; conv v_ext R2 0.535. T_ext MAE 38.1 C conv, 35.2 C genai.
- The plain ARD GP collapses on v_ext (MAPE 95-120%, R2 <= 0.08): the grid
  decode, not the GP itself, is the win.
- Calibration caveat: grid-GP coverage90 is around 0.5, below the 0.90
  nominal; point-accurate, under-dispersed. Report with the caveat.

## Voice and format rules

- Accessible, short paragraphs, figures carry the story. This is a workshop
  paper, not the dense main-track style.
- No em dashes.
- Title poses the question.
- References: red placeholders only, `\needref{topic}` renders
  [REF: topic] in red; the author wires the real bibliography manually.
  Exception: the two self-references (Acta 2025 published, NeurIPS 2026 under
  review) get real bib entries since they are load-bearing for positioning.
- Every number in the paper must trace to results/tables/composition_heads_cv.csv
  or process_heads_cv.csv (5-fold means +/- std).
- Template: neurips_2026.sty (not in the repo; drop it into paper/ai4mat/
  from the NeurIPS 2026 style kit before compiling). Preamble mirrors
  old_papers/Neurips_2026.tex.

## Files

- paper/ai4mat/manifesto.md (this file)
- paper/ai4mat/skeleton.md (section plan, page budget, figure map)
- paper/ai4mat/main.tex (the paper)
- paper/ai4mat/figures/ (fig1_missing_leg, fig2_pipeline, fig3_gnn)
