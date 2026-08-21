# Result figures: context and discussion

Companion notes for the figures under `paper/ai4mat/figures/`. Each entry
states what the figure shows, how it was produced, and what a reader should
take away. All figures use condition-grouped 5-fold CV; unless stated
otherwise they are drawn on `cv_fold0` for illustration, while the reported
metrics in the paper are pooled or averaged over all five folds.

## fig_temperature_parity.png

**What it is.** Predicted against true extrusion temperature, one panel per
representation branch (conventional, vision embedding). Purple markers are the
joint-grid GP (the scored value is the catalogue-decoded one; the vertical bar
is the pre-decode latent 90% interval), red markers are the GBDT. Circles /
triangles / squares are train / val / test conditions, offset slightly in x so
the two heads do not overlap. Grey vertical lines are the observed temperature
settings; the dashed diagonal is perfect prediction.

**How it was made.** Both heads were fitted on the fold's train conditions
only and applied to all three parts. The GP's scored output snaps to the
catalogue, so its markers sit on the discrete temperature grid by construction;
the uncertainty bar shows the continuous latent estimate underneath.

**What it shows.** Both heads track the diagonal but saturate at the high
end: above ~400 degC the predictions flatten near 450 even when the truth
reaches 500, the same contraction toward the training range documented for
velocity. The GP's 90% bars cover the diagonal on most test points except at
the extremes, where the interval is wide but still centred low. The GBDT
sits inside the GP's band nearly everywhere, so the two heads agree on the
shape of the response; they differ mainly in that the GP is honest about how
little it knows at the edges.

**Caveat.** A single fold is shown. The temperature grid is discrete, so
horizontal position is exact truth, not noise.

## fig_lda_temperature.png

**What it is.** Temperature response along a 1-D supervised projection of the
head input, one panel per branch. The projection is selected on the validation
part from five candidates (two LDA variants, PLS, ridge and lasso fits) by
Spearman correlation with the target, with a small tolerance favouring the
simpler regression axes. Points are conditions on the projected axis; the
purple curve and band are the kernel-smoothed joint-grid GP mean and 90%
interval, the red dashed curve the GBDT.

**How it was made.** Features are the per-condition pooled head inputs
(representation plus known composition), standardized on train. Curves use an
adaptive Gaussian-kernel mean (h = 0.12) that widens to reach 5 neighbours
where the axis is sparse. The heads see the full representation, so the curve
is a mean response along the axis, not the model's raw output; the scatter of
points around the curve is the variation the 1-D view cannot explain.

**What it shows.** Temperature is monotonic in the selected projection for
both branches (conventional chose a ridge fit, the vision embedding a lasso
fit), and the two heads' curves nearly coincide, so the representation itself,
not the head, sets the achievable response. The band widens at both ends where
few conditions lie. Held-out points (val, test) follow the same trend as
train, so the projection is not overfit to the training part.

**Caveat.** The selected axis is one of several defensible projections; its
validation Spearman is strong (~0.8) but the test correlation is lower
(~0.6-0.8), so this is a visualization aid, not a fitted model. Temperature
being a coarse discrete setting is why points form horizontal rows.

## fig_gnn_lda_velocity.png and fig_conventional_lda_velocity.png

**What it is.** Velocity response along a supervised projection of the head
input, shown raw (left) and in log space (right), for one branch per figure
(GNN or conventional). Same construction as the temperature projection
figure: points are conditions, the purple curve and band are the smoothed
joint-grid GP mean and 90% interval, the red dashed curve is the GBDT.

**How it was made.** The velocity projection is selected on validation from
the same five candidates and the axis is centred and scaled on train. The
GBDT is fitted in log space, so its raw-space curve is the exponential of the
smoothed log response. The GNN branch chose a ridge fit, the conventional
branch a lasso fit.

**What it shows.** The response is a clean sigmoid: flat at the low-velocity
end, a steep rise through the middle, and saturation at the high end. The GNN
branch's projection is better behaved than the conventional one, whose axis is
dominated by a single extreme point (the leftmost cluster at projection -3).
Log space spreads the low-velocity conditions and compresses the saturated
top, which is why the two heads agree more closely there. Both heads contract
toward the training mean at the extremes, consistent with the parity figure.

**Caveat.** Raw velocity is dominated by the few fast conditions, so the raw
panel exaggerates agreement; the log panel is the more informative of the two.
A single fold is shown.

## fig_gnn_velocity_parity.png and fig_conventional_velocity_parity.png

**What it is.** Predicted against true velocity, raw (left) and log (right),
one figure per branch. Purple markers are the joint-grid GP with its latent
90% interval as a vertical bar, red markers are the GBDT, and the grey
vertical lines are the observed press settings.

**How it was made.** Both heads were fitted on train and applied to all three
parts. The GP's scored value is catalogue-snapped and its bar is the
pre-snap latent interval; the GBDT is applied in log space and exponentiated
for the raw panel. Points for train, val and test are jittered slightly in x
so the two heads do not sit on top of each other.

**What it shows.** Both branches are tight at low velocity and spread at high
velocity, and the spread is asymmetric: the heads rarely overshoot a slow
condition but often undershoot a fast one, the contraction already seen in
temperature. The GP's bars widen with velocity and mostly still cover the
diagonal, so its uncertainty tracks the difficulty. The log panel makes the
low-velocity agreement legible and shows that most of the absolute error is
concentrated in the handful of fast conditions.

**Caveat.** Raw-space error is inflated by the log-space decoding of both
heads; the log panel is the fairer view of how the heads rank conditions. A
single fold is shown.

## fig_composition_taxonomy.png

**What it is.** Where the top-1 alloy call lands, per alloy, one panel per
branch. Each row is one of the 14 alloys ordered by family (the Mg-Gd(-Mn)
grid, then the dilute-Zn trio, then AZ31 and ME21); the stacked bar is the
fraction of that alloy's held-out conditions called exactly, off by one
lattice step, in the same family, or in another family. Pooled over all five
CV folds (107 conditions), so the top-1 rates match Table 3 exactly.

**How it was made.** One lattice step is one Gd or one Mn step inside the
Mg-Gd(-Mn) grid, or a swap within the Z1 / ZNd10 / ZX10 trio. Everything else
is same family or other family. The head is the OOF-constrained reranker,
which returns a lattice alloy, so "exact" and "one step" have a precise
geometric meaning here.

**What it shows.** The branches differ in concentration resolution, not in
chemistry recognition. Conventional is exact on AZ31 and ME21 and strong on
the dilute-Zn trio, and nearly all of its misses are one lattice step (within
one step 0.79 against a top-1 of 0.64). The embeddings drop to top-1 ~0.27 but
stay near 0.68 within one step, and "other family" is rare everywhere, so the
representation gap is in resolving Gd and Mn concentration, not in identifying
the alloy system.

**Caveat.** Rows have very different n (4 to 18), so a single misclassified
condition moves a small-n alloy's bar by 25%. The lattice-step category is
specific to this alloy set's grid structure.

## fig_composition_confidence.png

**What it is.** The probability the head assigns to the true alloy, one
raincloud per branch over the 107 pooled held-out conditions. The density is
above the line, one dot per condition below it (filled = top-1 correct, open =
top-1 wrong), the tick is the median, and the dashed line is chance (1/14).
Annotations reproduce the top-1 and NLL columns of Table 3 exactly.

**How it was made.** Confidence is continuous (106 of 107 distinct values per
branch), unlike the lattice-quantized element error, so it is the one Task A
quantity where a density is honest. The KDE is reflected at 0 and 1 and all
three rows share one height scale so the shapes are comparable.

**What it shows.** Conventional spreads probability broadly and is often
confidently right (median 0.40); both embeddings pile up near zero (median
~0.14), so they do not merely miss, they assign almost no probability to the
truth on much of the lattice, the mechanism behind their NLL near 2.3. Filled
and open dots separate around 0.3 to 0.4 for every branch, so confidence is a
usable rejection signal, which matters for a screening tool. The GNN has a
small bump back up near 1.0: it is either very sure and right or lost.

**Caveat.** Confidence and correctness are pooled over all 14 alloys, so the
strong easy alloys (AZ31, ME21) and the hard grid alloys share one curve.

## fig_composition_elements.png

**What it is.** Per-element error, two panels. Left: the per-element WAPE
where the element is present, split at zero into the share from
under-predicting and from over-predicting, so bar length is the metric and the
balance shows its direction; a "0" marker means the branch was exactly right
on every condition containing that element. Right: the presence
false-positive rate, the guard metric paired with WAPE in Table 3.

**How it was made.** Pooled over the 107 held-out conditions. The head decodes
to a lattice alloy, so per-element error takes only a few exact values, which
is why bars are used rather than a density. The pooled WAPE (13.4 / 41.8 /
43.4%) reproduces Table 3's per-element WAPE (13.6 / 41.6 / 44.2%) up to the
pooling-versus-fold-averaging difference.

**What it shows.** Conventional is exactly right on Zn, Al, Ce and Y, so its
whole error budget sits in Gd, Mn, Nd and Ca, the elements that vary within a
family. Its Gd error leans to over-prediction, the low-Gd alloys pulled toward
the middle of the ladder. The embeddings under-predict the rare elements
(Nd WAPE ~87%, Ca ~73-82%, nearly all under-prediction), meaning they miss the
alloys that contain them, while over-predicting Mn and Ca where they are
absent (up to 21% false positives on Mn).

**Caveat.** WAPE conflates how often an element is wrong with by how much; a
branch that is slightly off on every condition and one that is badly off on a
few can tie. The present / absent split is a property of this alloy lattice,
not a general composition benchmark.


