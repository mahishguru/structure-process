"""OOF-constrained reranker: convex blend of an owner head and a
complementary auxiliary head,

    p = owner_weight * p_owner + (1 - owner_weight) * p_aux

with the weight selected only from training out-of-fold (OOF) predictions. A
candidate weight is eligible only when its OOF all-element WAPE and
false-positive rate do not exceed the owner OOF baseline (the guard
constraints). Validation and test select nothing.

The owner is the condition-balanced CatBoost + kNN fusion. The auxiliary is
representation-dependent:
- conventional: uniform fusion of per-descriptor-block logistic heads on
  condition-mean features (the descriptor blocks from the pipeline builder);
- genai / gnn:  FT-Transformer classifier probabilities (these
  representations have no descriptor-block structure).

Owner hyperparameters are frozen by the caller (val-selected) before OOF
construction; the blend weight is the only OOF-selected quantity. Preprocessing
is refitted inside every inner fold for both owner and auxiliary.
"""

from __future__ import annotations

import numpy as np

from icme_mg.data import (ALLOYS, cls_targets, condition_pool,
                          conditioned_inputs)
from icme_mg.evaluation.head_metrics import composition_metrics
from icme_mg.heads.composition import ftt as ftt_head
from icme_mg.heads.composition import fusion
from icme_mg.protocols.cv import condition_stratified_splits

# Conventional descriptor blocks (descriptor width 438; the trailing four
# columns of the composition input are the condition-constant known process
# and ratio features).
CONVENTIONAL_BLOCKS = {
    "3point_PCA": (0, 120),
    "gram_Isomap": (120, 320),
    "aspect_ratio": (320, 349),
    "equivalent_diameter": (349, 378),
    "GSH_Isomap": (378, 438),
}
KNOWN_COLUMNS = 4  # known T_ext, log v_ext, ratio one-hot x2

DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_OWNER_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FTT_PCA_WIDTH = 512  # compress aux FTT inputs wider than this


def _align(reference, conditions, probabilities):
    lookup = {c: probabilities[i] for i, c in enumerate(conditions)}
    missing = [c for c in reference if c not in lookup]
    if missing:
        raise RuntimeError(f"probability alignment misses {missing}")
    return np.stack([lookup[c] for c in reference])


# --------------------------------------------------------------------- owner
def _owner_oof(raw_train, condition_order, inner_splits, cat_params,
               knn_params, log):
    """Strict OOF owner probabilities with preprocessing refit per fold."""
    X_raw, c_raw = raw_train
    c_raw = np.asarray(c_raw).astype(str)
    condition_array = np.asarray(condition_order)
    output = np.zeros((len(condition_order), len(ALLOYS)))
    filled = np.zeros(len(condition_order), dtype=bool)
    for fold, (fit_idx, held_idx) in enumerate(inner_splits):
        fit_conditions = condition_array[fit_idx]
        held_conditions = condition_array[held_idx]
        fit_mask = np.isin(c_raw, fit_conditions)
        held_mask = np.isin(c_raw, held_conditions)
        inner_raw = ((X_raw[fit_mask], c_raw[fit_mask]),
                     (X_raw[held_mask], c_raw[held_mask]))
        inner_parts, _ = conditioned_inputs(inner_raw)
        log(f"    owner OOF fold {fold + 1}/{len(inner_splits)}: "
            f"fit={len(fit_conditions)} held={len(held_conditions)}")
        model = fusion.fit_balanced_fusion(
            *inner_parts[0], cat_params=cat_params, knn_params=knn_params)
        raw_probability = fusion.fusion_probabilities(
            model, inner_parts[1][0])
        pooled_conditions, pooled = condition_pool(
            raw_probability, inner_parts[1][1])
        output[held_idx] = _align(
            list(held_conditions), list(pooled_conditions),
            np.asarray(pooled))
        filled[held_idx] = True
    if not filled.all():
        raise RuntimeError("owner OOF did not predict every condition")
    return output


# ---------------------------------------------------------- block auxiliary
def _block_features(features, condition_ids, columns):
    """One row per condition: block descriptor mean + known columns."""
    condition_ids = np.asarray(condition_ids).astype(str)
    conditions = sorted(set(condition_ids))
    start, stop = columns
    known_start = features.shape[1] - KNOWN_COLUMNS
    rows = []
    for condition in conditions:
        bag = features[condition_ids == condition]
        rows.append(np.concatenate([
            bag[:, start:stop].mean(axis=0),
            bag[:, known_start:].mean(axis=0),
        ]))
    return conditions, np.stack(rows)


def _make_block_classifier(C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                           random_state=seed))


def _aligned_probabilities(model, features):
    """Pipeline predict_proba expanded to the full alloy lattice."""
    raw = np.asarray(model.predict_proba(features))
    output = np.zeros((len(features), len(ALLOYS)))
    output[:, np.asarray(model[-1].classes_, dtype=int)] = raw
    return output


def _fit_blocks(train_part, inner_splits, c_values, seed, log):
    """Per-block logistic heads with inner-CV selection of C. Returns
    (final_models, uniform_oof, selected_c, search_rows)."""
    features, condition_ids = train_part
    models, oof_blocks, selected_c, search_rows = {}, {}, {}, []
    order = None
    for block_index, (block, columns) in enumerate(
            CONVENTIONAL_BLOCKS.items()):
        conditions, values = _block_features(features, condition_ids, columns)
        if order is None:
            order = conditions
        elif conditions != order:
            raise RuntimeError("block condition orders differ")
        y = cls_targets(np.asarray(conditions))
        cache = {}
        scores = {}
        for C in c_values:
            oof = np.zeros((len(y), len(ALLOYS)))
            for fold, (fit_idx, held_idx) in enumerate(inner_splits):
                model = _make_block_classifier(C, seed + block_index + fold)
                model.fit(values[fit_idx], y[fit_idx])
                oof[held_idx] = _aligned_probabilities(
                    model, values[held_idx])
            report = composition_metrics(oof, np.asarray(conditions))
            search_rows.append({
                "block": block, "C": C,
                **{k: v for k, v in report.items() if k != "per_element"},
            })
            cache[C] = oof
            scores[C] = report
        best_c = min(
            c_values,
            key=lambda C: (scores[C]["element_wape_present"],
                           scores[C]["element_wape_all"],
                           scores[C]["element_mae_macro"],
                           scores[C]["class_nll"], C))
        log(f"    block {block}: selected C={best_c}")
        model = _make_block_classifier(best_c, seed + block_index)
        model.fit(values, y)
        models[block] = model
        oof_blocks[block] = cache[best_c]
        selected_c[block] = best_c
        for row in search_rows:
            if row["block"] == block:
                row["selected"] = row["C"] == best_c
    uniform_oof = np.mean(
        [oof_blocks[block] for block in CONVENTIONAL_BLOCKS], axis=0)
    return models, uniform_oof, selected_c, search_rows, order


def _predict_blocks(models, part):
    features, condition_ids = part
    combined = None
    conditions = None
    for block, columns in CONVENTIONAL_BLOCKS.items():
        block_conditions, values = _block_features(
            features, condition_ids, columns)
        probabilities = _aligned_probabilities(models[block], values)
        if combined is None:
            combined = probabilities
            conditions = block_conditions
        else:
            if block_conditions != conditions:
                raise RuntimeError("block condition orders differ")
            combined += probabilities
    return conditions, combined / len(CONVENTIONAL_BLOCKS)


# ------------------------------------------------------------ FTT auxiliary
def _maybe_compress(parts, log):
    if parts[0][0].shape[1] > FTT_PCA_WIDTH:
        compressed, n = ftt_head.pca_compress_parts(parts)
        log(f"    aux FTT input PCA-compressed to {n} components")
        return compressed
    return parts


def _ftt_oof(raw_train, condition_order, inner_splits, seed, log):
    X_raw, c_raw = raw_train
    c_raw = np.asarray(c_raw).astype(str)
    condition_array = np.asarray(condition_order)
    output = np.zeros((len(condition_order), len(ALLOYS)))
    filled = np.zeros(len(condition_order), dtype=bool)
    for fold, (fit_idx, held_idx) in enumerate(inner_splits):
        fit_conditions = condition_array[fit_idx]
        held_conditions = condition_array[held_idx]
        fit_mask = np.isin(c_raw, fit_conditions)
        held_mask = np.isin(c_raw, held_conditions)
        inner_raw = ((X_raw[fit_mask], c_raw[fit_mask]),
                     (X_raw[held_mask], c_raw[held_mask]))
        inner_parts, _ = conditioned_inputs(inner_raw)
        inner_parts = _maybe_compress(inner_parts, log)
        log(f"    aux FTT OOF fold {fold + 1}/{len(inner_splits)}")
        model = ftt_head.fit_ftt_classifier(*inner_parts[0])
        raw_probability = ftt_head.predict_ftt_classifier(
            model, inner_parts[1][0])
        pooled_conditions, pooled = condition_pool(
            raw_probability, inner_parts[1][1])
        output[held_idx] = _align(
            list(held_conditions), list(pooled_conditions),
            np.asarray(pooled))
        filled[held_idx] = True
    if not filled.all():
        raise RuntimeError("aux FTT OOF did not predict every condition")
    return output


# ----------------------------------------------------------- weight selection
def select_owner_weight(condition_order, owner_oof, aux_oof, owner_weights,
                        all_wape_tolerance, false_positive_tolerance):
    """Constrained lexicographic selection on training OOF predictions."""
    conditions = np.asarray(condition_order)
    owner_report = composition_metrics(owner_oof, conditions)
    rows = []
    cache = {}
    for weight in sorted(set(owner_weights)):
        probability = weight * owner_oof + (1.0 - weight) * aux_oof
        report = composition_metrics(probability, conditions)
        all_ok = (report["element_wape_all"]
                  <= owner_report["element_wape_all"]
                  + all_wape_tolerance + 1e-12)
        fp_ok = (report["macro_false_positive_rate"]
                 <= owner_report["macro_false_positive_rate"]
                 + false_positive_tolerance + 1e-12)
        rows.append({
            "owner_weight": weight,
            "eligible": bool(all_ok and fp_ok),
            "all_wape_constraint_pass": bool(all_ok),
            "false_positive_constraint_pass": bool(fp_ok),
            **{k: v for k, v in report.items() if k != "per_element"},
        })
        cache[weight] = probability
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no eligible reranking weight; owner_weight=1 must always "
            "satisfy the zero-tolerance owner constraints")
    # Composition WAPE is discrete after the nominal-alloy argmax; prefer the
    # more conservative owner-heavy candidate when metrics tie.
    selected_row = sorted(
        eligible,
        key=lambda row: (
            row["element_wape_present"], row["element_wape_all"],
            row["macro_false_positive_rate"], row["element_mae_macro"],
            -row["owner_weight"], row["class_nll"]))[0]
    selected = float(selected_row["owner_weight"])
    for row in rows:
        row["selected"] = bool(np.isclose(row["owner_weight"], selected))
    return selected, cache[selected], rows


# ------------------------------------------------------------------- driver
def run_reranker(raw_parts, aux, *, inner_folds=5,
                 c_values=DEFAULT_C_VALUES,
                 owner_weights=DEFAULT_OWNER_WEIGHTS,
                 all_wape_tolerance=0.0, false_positive_tolerance=0.0,
                 seed=0, owner_cat_params=None, owner_knn_params=(5, 0.05),
                 log=print):
    """Fit owner + auxiliary with strict training OOF and predict val/test.

    aux: "blocks" (conventional) or "ftt" (genai, gnn).
    Returns condition-level probabilities per part for owner, auxiliary, and
    the reranked blend, plus the full selection audit trail.
    """
    if aux not in ("blocks", "ftt"):
        raise ValueError(f"unknown auxiliary {aux!r}")
    if aux == "blocks" and raw_parts[0][0].shape[1] != 438:
        raise ValueError(
            "blocks auxiliary requires the 438-D conventional descriptor "
            f"space, got {raw_parts[0][0].shape[1]}")
    if not any(np.isclose(w, 1.0) for w in owner_weights):
        raise ValueError("owner_weights must contain 1.0 (safe fallback)")

    parts, _ = conditioned_inputs(raw_parts)
    train_conditions = sorted(set(np.asarray(parts[0][1]).astype(str)))
    inner_splits, protocol, actual_folds = condition_stratified_splits(
        train_conditions, inner_folds, seed)

    log(f"  reranker({aux}): owner OOF over {actual_folds} inner folds")
    owner_oof = _owner_oof(
        raw_parts[0], train_conditions, inner_splits,
        owner_cat_params, owner_knn_params, log)

    if aux == "blocks":
        aux_models, aux_oof, selected_c, search_rows, _ = _fit_blocks(
            parts[0], inner_splits, c_values, seed + 20, log)
    else:
        aux_oof = _ftt_oof(
            raw_parts[0], train_conditions, inner_splits, seed + 20, log)
        selected_c, search_rows, aux_models = None, [], None

    owner_weight, reranked_oof, weight_table = select_owner_weight(
        train_conditions, owner_oof, aux_oof, owner_weights,
        all_wape_tolerance, false_positive_tolerance)
    log(f"  reranker({aux}): selected owner weight={owner_weight:.2f}")

    owner = fusion.fit_balanced_fusion(
        *parts[0], cat_params=owner_cat_params, knn_params=owner_knn_params)
    ftt_full, ftt_parts = None, None
    if aux == "ftt":
        ftt_parts = _maybe_compress(parts, log)
        ftt_full = ftt_head.fit_ftt_classifier(*ftt_parts[0])
    probabilities = {"train_oof_owner": owner_oof, "train_oof_aux": aux_oof,
                     "train_oof_reranked": reranked_oof}
    conditions = {"train": np.asarray(train_conditions, dtype=object)}
    for part_index, part in ((1, "val"), (2, "test")):
        pooled_conditions, owner_pooled = condition_pool(
            fusion.fusion_probabilities(owner, parts[part_index][0]),
            parts[part_index][1])
        if aux == "blocks":
            aux_conditions, aux_pooled = _predict_blocks(
                aux_models, parts[part_index])
        else:
            aux_conditions, aux_pooled = condition_pool(
                ftt_head.predict_ftt_classifier(
                    ftt_full, ftt_parts[part_index][0]),
                parts[part_index][1])
        aux_pooled = _align(
            list(pooled_conditions), list(aux_conditions), aux_pooled)
        reranked = owner_weight * owner_pooled \
            + (1.0 - owner_weight) * aux_pooled
        probabilities[f"{part}_owner"] = np.asarray(owner_pooled)
        probabilities[f"{part}_aux"] = np.asarray(aux_pooled)
        probabilities[f"{part}_reranked"] = np.asarray(reranked)
        conditions[part] = np.asarray(pooled_conditions, dtype=object)

    return {
        "probabilities": probabilities,
        "conditions": conditions,
        "selected_owner_weight": owner_weight,
        "weight_table": weight_table,
        "block_selected_c": selected_c,
        "block_search": search_rows,
        "inner_cv_protocol": protocol,
        "inner_folds": actual_folds,
        "aux": aux,
    }
