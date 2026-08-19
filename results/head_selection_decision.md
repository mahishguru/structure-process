# Head selection and repo consolidation

Decision record comparing `main` and `winners`. Source: `results/winner_benchmark.json`
(winners), `results/tables/*.csv` (main), and the eight `scripts/louam_*.py` winner
implementations.

## 1. The finding that drives every decision

`element_wape_present` is the paper's headline metric by design: in this
campaign the alloy family is known once an optimized microstructure exists, so
only the levels of the present elements must be predicted. It is still gameable
in isolation - a head that predicts every element as present scores well on it
while being useless. Five of the eight winner rows did exactly that, which is
why every head now also reports all-element WAPE and false-positive rate as
guard columns:

| pipeline | head | folds | present WAPE | all WAPE | FP rate | top-1 |
|---|---|---:|---:|---:|---:|---:|
| conventional | OOF-constrained reranker | 5 | **11.5%** | **15.3%** | **0.4%** | **0.644** |
| conventional | confidence gate + presence guard | 1 | 17.0% | 29.6% | 2.5% | 0.500 |
| gnn | LMNN-8D + hurdle/level | 1 | 17.0% | 67.2% | 14.0% | - |
| genai | PCA8 chain presence + posterior level | 1 | 20.2% | 60.7% | 12.3% | - |
| genai | raw latent + hurdle/level | 1 | 25.9% | 72.7% | 13.2% | - |
| gnn | LMNN-8D + kNN | 5 | 30.2% | 49.9% | 4.8% | 0.417 |
| genai | LMNN-8D + balanced fusion | 5 | 36.5% | 112.1% | 18.7% | 0.358 |
| genai | LMNN-8D + hurdle/level | 1 | 37.0% | 165.2% | 36.6% | - |

Only the conventional reranker improves all four metrics at once. The hurdle-style
GNN and GenAI "winners" buy present-WAPE with false positives.

`main` does not measure all-element WAPE or false-positive rate at all
(`results/tables/composition_heads.csv` has 8 columns, neither is among them).
That gap must close before any head is called a winner.

## 2. Protocol

`cv_fold0..4` are condition-grouped 5-fold partitions (every test alloy
also appears in that fold's training set). This is the same estimand as
`random_seed0`, with 5 folds instead of 1.

Consequence: 5-fold is strictly the better report of the same claim (107 conditions
covered instead of 18, plus dispersion). Single-split seed-0 numbers move in steps of
0.056 on top-1 and cannot separate heads.

**Decision: 5-fold cross-validation is the only reported protocol. It is
called "5-fold cross-validation" in the paper and everywhere else.** The split
files are named `cv_fold*` on disk and the protocol label in all reports is
`condition_grouped_5fold_cv`.

## 3. Task A (composition given known process): head set

Winner: **conventional + OOF-constrained reranker**, 11.5% +/- 2.0% present-WAPE over
5 folds, against its matched 5-fold baseline of 16.3% +/- 3.6%.

Three heads, evaluated on all three representations:

| # | head | role |
|---|---|---|
| 1 | kNN retrieval | retrieval baseline; shows the 14-alloy lattice is the right prior |
| 2 | Condition-balanced fusion (CatBoost + kNN) | previous headline; the comparator the winner must beat |
| 3 | OOF-constrained reranker | winner |

Dropped from the paper table: XGBoost classifier, CatBoost classifier alone,
FT-Transformer, prior-adjusted fusion, XGBoost regression, CatBoost regression.
They are comparators that neither win nor carry an argument.

Kept as a short negative result, not as winners: LMNN-8D + hurdle/level. It is the
cleanest demonstration of the metric trap in section 1 and is worth two sentences.

## 4. Task B (process given known composition): head set

| # | head | role |
|---|---|---|
| 1 | Gradient-boosted trees (XGBoost or CatBoost, one of them) | point-prediction baseline |
| 2 | GP, ARD-RBF + lognormal smearing | calibrated head; carries coverage90 |
| 3 | Condition-distribution joint-grid GP (T_ext) + OOF-constrained log-GP/ordinal (v_ext) | winner |

The Task B winner is conventional for both targets: T_ext MAE 30.6 C / MAPE 8.6%,
v_ext MAPE 40.3% / R2 0.651. Both are single-split seed-0 only and must be re-run
under 5-fold before they can be reported.

The winner decodes over the discrete grid of observed (T, log v) training pairs
rather than regressing continuously, which is why it beats the continuous heads:
the process space is a small discrete catalogue, exactly like the composition lattice.
That parallel is the paper's cleanest structural argument and should be stated.

Both winner heads already produce conformal 90% intervals fitted on OOF residuals, so
coverage stays reportable.

## 5. Representation verdict

Conventional descriptors win both tasks. GNN and GenAI survive as the comparison that
makes the claim meaningful, not as winners. This is a sharper story than the current
"different halves favour different representations", which the rebalanced split already
weakened.

## 6. What each branch must take from the other

`winners` -> `main`:
- 5-fold protocol as headline
- all-element WAPE and macro false-positive rate as guard metrics
- OOF-constrained reranker (Task A), joint-grid GP and ordinal velocity (Task B)
- nested selection discipline: candidates ranked on inner-fold OOF, never on val or test

`main` -> `winners`:
- per-pipeline val-selected hyperparameters for XGBoost, CatBoost, kNN
- ARD-RBF GP kernel and the lognormal smearing correction
- PCA compression of the GenAI input for the FT-Transformer
- the in-support split rebalance, which matters less under 5-fold since folds rotate,
  but still fixes the seed-0 appendix rows

## 7. Blockers before anything is reported

1. **Element lattice.** Resolved by decision: the lattice stays at 8 elements
   (Al, Zn, Mn, Ce, Gd, Ca, Nd, Y), matching winners. Main's tuned seed-0 numbers were
   computed at 7 and must be re-run at 8 before they enter any table.
2. **Post-hoc design.** The 11.5% reranker was designed after inspecting these exact
   folds; its own docstring says "results are exploratory". The planned re-run on a
   fresh fold seed (step 5 below) is what turns it from a diagnostic into a claim.
3. **Obfuscated code.** `louam_genai_task_b_velocity_uncertainty_loco.py` and
   `louam_genai_wape_targeted_seed0.py` carry base85/zlib-compressed modules inlined as
   string blobs. Resolved by scope: neither head is in the selected sets, so the blobs
   are not ported. They remain in the `winners`/`louam` branch history only.

The GNN 5-fold rows keep the frozen seed-0 encoder rather than refitting it per fold.
Accepted: they are reported as fixed-encoder diagnostics.

## 7b. Inner-CV granularity fix (neurips-ai4mat)

The original winner code capped stratified inner folds at the smallest alloy
condition count. On the 5-fold partitions several Mg-Gd-Mn alloys have only two
training conditions, collapsing the OOF selection to 2 inner folds with a 50%
holdout. The branch now stratifies only when every alloy has at least the
requested number of conditions, and otherwise falls back to unstratified
condition-level KFold at full granularity (5 inner folds). Condition grouping,
which is what OOF honesty requires, is preserved in both paths.

## 8. Layout (executed on neurips-ai4mat)

```
src/icme_mg/
  data/__init__.py        labels, lattice, split loaders, conditioning contracts
  heads/
    composition/          knn.py, fusion.py, ftt.py, reranker.py
    process/              trees.py, gp.py, grid_gp.py
  evaluation/
    head_metrics.py       all metrics incl. all-WAPE + FP guards, calibration
    metrics.py            flow-head training metrics (pre-existing, untouched)
  protocols/
    cv.py                 5-fold CV driver constants, inner condition splits

scripts/
  run/
    run_cv.py             unified runner: folds x representations x head sets
    aggregate_cv.py       folds -> results/tables/*_cv.csv
  (pre-existing scripts remain for reference; the runner imports only
  src/icme_mg, no cross-script imports)

results/
  tables/                 generated CSVs only
runs/cv/{fold}/{pipeline}/  report.json + predictions.npz (raw predictions, so
                            any new metric is a recomputation, never a re-fit)
```

Head interface: composition heads consume conditioned parts and produce alloy
probabilities; process heads consume conditioned parts (trees, gp) or raw parts
(grid_gp bags) and produce physical-unit (T_ext, v_ext). The reranker is one
selection machinery with a representation-appropriate auxiliary: descriptor
blocks for conventional, FT-Transformer for genai/gnn.

## 9. Order of work

1. Port the guard metrics into evaluation. Done on neurips-ai4mat
   (`evaluation/head_metrics.py`).
2. Restore the 8-element lattice. Done: Y column added to labels.csv
   (ME21 0.04, all others 0), source switched to the winners xlsx.
3. Rebuild the 5-fold runner over all three representations and both head
   sets. Done (`scripts/run/run_cv.py`, smoke-tested on fold 0).
4. Obfuscated GenAI scripts: not needed, none of their heads are selected.
5. Full re-run: 5 folds x 3 representations x all heads, fresh, no previous
   results. Then `aggregate_cv.py` produces the paper tables.
6. If the reranker holds across folds, it is the headline; if not, the
   condition-balanced fusion stays.
