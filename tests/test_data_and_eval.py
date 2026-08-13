"""Data-layer tests: labels csv integrity, split hygiene, sampler balance,
metrics and calibration sanity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icme_mg import ELEMENTS, LABEL_ORDER
from icme_mg.evaluation.calibration import pit_values, quantile_ece
from icme_mg.evaluation.metrics import (aggregate_per_condition, alloy_topk,
                                        per_label_metrics, presence_f1)
from icme_mg.training.datasets import BalancedConditionSampler
from icme_mg.training.splits import build_random_split

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels" / "labels.csv"
SPLITS = ROOT / "data" / "splits"

requires_data = pytest.mark.skipif(not LABELS.exists(),
                                   reason="labels.csv not built")


@requires_data
class TestLabelsCSV:
    def test_columns_and_ranges(self):
        df = pd.read_csv(LABELS)
        for col in ["condition_id", "alloy", "base_alloy",
                    "extrusion_ratio_type"] + LABEL_ORDER:
            assert col in df.columns
        assert df["condition_id"].is_unique
        assert (df[ELEMENTS] >= 0).all().all()
        assert df["T_ext"].between(150, 550).all()
        assert (df["v_ext"] > 0).all()

    def test_extrusion_ratio_types(self):
        df = pd.read_csv(LABELS)
        is_mg_gd = (df["base_alloy"].str.startswith("Mg-")
                    & df["base_alloy"].str.contains("Gd"))
        assert set(df["extrusion_ratio_type"]) == {
            "standard", "mg_gd_series"}
        assert (df.loc[is_mg_gd, "extrusion_ratio_type"]
                == "mg_gd_series").all()
        assert (df.loc[~is_mg_gd, "extrusion_ratio_type"]
                == "standard").all()

    def test_expected_counts(self):
        df = pd.read_csv(LABELS)
        assert len(df) == 108
        assert df["base_alloy"].nunique() == 14


@requires_data
class TestSplits:
    def test_primary_split_files_valid(self):
        df = pd.read_csv(LABELS)
        all_ids = set(df["condition_id"])
        s = json.loads((SPLITS / "random_seed0.json").read_text())
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert tr | va | te == all_ids
        assert not (tr & va or tr & te or va & te)
        assert s["seed"] == 0

    def test_grouped_random_protocol(self):
        df = pd.read_csv(LABELS)
        splits = build_random_split(
            df, val_fraction=0.15, test_fraction=0.15, seed=0)
        assert set(splits) == {"random_seed0"}
        split = splits["random_seed0"]
        assert split["protocol"] == "alloy_stratified_random_condition_split"
        train_alloys = set(df.loc[
            df["condition_id"].isin(split["train"]), "base_alloy"])
        assert train_alloys == set(df["base_alloy"])


class TestSampler:
    def test_balance(self):
        cids = ["a"] * 50 + ["b"] * 3 + ["c"] * 1
        s = BalancedConditionSampler(cids, samples_per_condition=8, seed=0)
        idx = list(iter(s))
        assert len(idx) == 24
        counts = {"a": 0, "b": 0, "c": 0}
        for i in idx:
            counts[cids[i]] += 1
        assert counts == {"a": 8, "b": 8, "c": 8}

    def test_epoch_changes_draws(self):
        cids = ["a"] * 10 + ["b"] * 10
        s = BalancedConditionSampler(cids, 4, seed=0)
        e0 = list(iter(s))
        s.set_epoch(1)
        e1 = list(iter(s))
        assert e0 != e1


class TestMetrics:
    def test_per_label_perfect(self):
        y = np.abs(np.random.default_rng(0).normal(2, 1, (20, 10))) + 0.1
        m = per_label_metrics(y, y)
        for name in LABEL_ORDER:
            assert m[name]["mae"] == 0

    def test_presence_f1_perfect(self):
        p = np.random.default_rng(0).random((30, 8)) > 0.5
        assert presence_f1(p, p)["macro"] == pytest.approx(1.0)

    def test_aggregate(self):
        cids = ["x", "x", "y"]
        v = np.array([[1.0], [3.0], [5.0]])
        out_ids, agg = aggregate_per_condition(cids, v)
        assert out_ids == ["x", "y"]
        assert agg.tolist() == [[2.0], [5.0]]

    def test_alloy_topk_selfmatch(self):
        rng = np.random.default_rng(0)
        nominal = rng.uniform(0, 5, (4, len(ELEMENTS)))
        names = ["A", "B", "C", "D"]
        y = np.concatenate([nominal, rng.uniform(200, 400, (4, 2))], axis=1)
        res = alloy_topk(y, nominal, names, names)
        assert res["top1"] == 1.0


class TestCalibration:
    def test_pit_uniform_for_perfect_model(self):
        rng = np.random.default_rng(0)
        B, S = 200, 400
        samples = rng.normal(0, 1, (B, S, 10)) + 5  # keep positive
        y_true = rng.normal(0, 1, (B, 10)) + 5
        pit = pit_values(y_true, samples)
        ece = quantile_ece(pit)
        assert ece["macro"] < 0.05
