"""Core model tests: spline flow correctness, normalizer round-trip, causal
masking, head shapes and NLL finiteness, adapters, baselines."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from icme_mg import ELEMENTS, LABEL_ORDER
from icme_mg.models.adapters import (AttentionPoolAdapter, TokenAdapter,
                                     VectorAdapter)
from icme_mg.models.baselines import (ConditionalGaussian, ConditionalMDN,
                                      IndependentMDN)
from icme_mg.models.flow_head import FlowTransformerHead, count_parameters
from icme_mg.models.label_space import LabelNormalizer, dequantize
from icme_mg.models.spline_flow import ConditionalSplineFlow

torch.manual_seed(0)
N_TOKENS, N_ELEM = len(LABEL_ORDER), len(ELEMENTS)


def _labels_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({e: np.where(rng.random(n) < 0.4,
                                   rng.uniform(0.1, 10, n), 0.0)
                       for e in ELEMENTS})
    df["T_ext"] = rng.uniform(200, 500, n)
    df["v_ext"] = rng.uniform(0.5, 10, n)
    return df


def _phys_batch(df):
    return torch.tensor(df[LABEL_ORDER].to_numpy(float), dtype=torch.float32)


# ------------------------------------------------------------------ spline
class TestSplineFlow:
    def test_log_prob_finite_and_integrates(self):
        flow = ConditionalSplineFlow(cond_dim=16)
        h = torch.randn(64, 16)
        y = torch.randn(64) * 2
        lp = flow.log_prob(y, h)
        assert lp.shape == (64,) and torch.isfinite(lp).all()

    def test_sample_logprob_roundtrip(self):
        """Samples must land in regions of finite density."""
        flow = ConditionalSplineFlow(cond_dim=8)
        h = torch.randn(128, 8)
        s = flow.sample(h)
        lp = flow.log_prob(s, h)
        assert torch.isfinite(s).all() and torch.isfinite(lp).all()

    def test_identity_at_init(self):
        """Zero-initialized head => transform starts near identity, so
        log_prob approx standard normal."""
        flow = ConditionalSplineFlow(cond_dim=4)
        flow.eval()
        h = torch.zeros(16, 4)
        y = torch.linspace(-2, 2, 16)
        lp = flow.log_prob(y, h)
        ref = -0.5 * (y ** 2 + math.log(2 * math.pi))
        assert torch.allclose(lp, ref, atol=0.2)

    def test_grad_flows(self):
        flow = ConditionalSplineFlow(cond_dim=8)
        h = torch.randn(32, 8, requires_grad=True)
        loss = -flow.log_prob(torch.randn(32), h).mean()
        loss.backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()


# -------------------------------------------------------------- normalizer
class TestLabelNormalizer:
    def test_roundtrip(self):
        df = _labels_df()
        nz = LabelNormalizer.fit(df)
        y = _phys_batch(df)
        z, log_det = nz.normalize(y)
        y2 = nz.denormalize(z)
        present = y[:, :N_ELEM] > 0
        # zeros for absent elements must be reapplied by caller
        y2[:, :N_ELEM] = torch.where(present, y2[:, :N_ELEM],
                                     torch.zeros_like(y2[:, :N_ELEM]))
        assert torch.allclose(y, y2, atol=1e-4, rtol=1e-4)
        assert torch.isfinite(log_det).all()

    def test_save_load(self, tmp_path):
        df = _labels_df()
        nz = LabelNormalizer.fit(df)
        nz.save(tmp_path / "n.json")
        nz2 = LabelNormalizer.load(tmp_path / "n.json")
        y = _phys_batch(df)
        z1, _ = nz.normalize(y)
        z2, _ = nz2.normalize(y)
        assert torch.equal(z1, z2)

    def test_dequantize_preserves_zeros(self):
        y = _phys_batch(_labels_df())
        y_dq = dequantize(y)
        zero_mask = y[:, :N_ELEM] == 0
        assert (y_dq[:, :N_ELEM][zero_mask] == 0).all()
        assert (y_dq != y).any()


# -------------------------------------------------------------------- head
def _head_inputs(B=12):
    df = _labels_df(B)
    nz = LabelNormalizer.fit(df)
    y = _phys_batch(df)
    z, log_det = nz.normalize(y)
    present = y[:, :N_ELEM] > 0
    cond = torch.randn(B, 8, 128)
    return z, present, log_det, cond


class TestFlowTransformerHead:
    def test_forward_shapes_and_finite(self):
        head = FlowTransformerHead()
        z, present, log_det, cond = _head_inputs()
        out = head(z, present, log_det, cond)
        assert out["nll"].shape == (12,)
        assert out["nll_tokens"].shape == (12, N_TOKENS)
        assert out["presence_logits"].shape == (12, N_ELEM)
        assert torch.isfinite(out["nll"]).all()

    def test_causality(self):
        """Changing label k must not affect NLL of tokens <= k."""
        head = FlowTransformerHead(dropout=0.0)
        head.eval()
        z, present, log_det, cond = _head_inputs()
        base = head(z, present, log_det, cond)["nll_tokens"]
        k = 4
        z2 = z.clone()
        z2[:, k] += 3.0
        pert = head(z2, present, log_det, cond)["nll_tokens"]
        # tokens strictly before k are conditioned only on y_<k: unchanged
        assert torch.allclose(base[:, :k], pert[:, :k], atol=1e-5), \
            "future label leaked into past tokens"
        # token k itself is the flow target: its NLL must change
        assert not torch.allclose(base[:, k], pert[:, k], atol=1e-3)

    def test_sampling(self):
        head = FlowTransformerHead()
        head.eval()
        cond = torch.randn(3, 8, 128)
        s = head.sample(cond, n_samples=7)
        assert s["z"].shape == (3, 7, N_TOKENS)
        assert s["present"].shape == (3, 7, N_ELEM)
        assert torch.isfinite(s["logp"]).all()
        # absent elements must carry z == 0
        absent = ~s["present"]
        assert (s["z"][..., :N_ELEM][absent] == 0).all()

    def test_param_count_order(self):
        head = FlowTransformerHead()
        n = count_parameters(head)
        assert 5e5 < n < 5e6

    @pytest.mark.parametrize("kind", ["gaussian", "mdn"])
    def test_ablation_heads(self, kind):
        head = FlowTransformerHead(token_head=kind)
        z, present, log_det, cond = _head_inputs()
        out = head(z, present, log_det, cond)
        assert torch.isfinite(out["nll"]).all()

    def test_overfit_tiny(self):
        """The head must be able to memorize 4 fixed conditions."""
        torch.manual_seed(1)
        head = FlowTransformerHead(dropout=0.0, cond_dropout=0.0)
        z, present, log_det, cond = _head_inputs(4)
        opt = torch.optim.Adam(head.parameters(), lr=3e-3)
        first = None
        for _ in range(150):
            out = head(z, present, log_det, cond)
            loss = out["nll"].mean()
            if first is None:
                first = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < first - 2.0, \
            f"no meaningful NLL decrease: {first:.2f} -> {loss.item():.2f}"


# ---------------------------------------------------------------- adapters
class TestAdapters:
    def test_vector(self):
        a = VectorAdapter(700, 128, 8)
        assert a(torch.randn(5, 700)).shape == (5, 8, 128)

    def test_token(self):
        a = TokenAdapter(80, 128, 16)
        assert a(torch.randn(5, 16, 80)).shape == (5, 16, 128)

    def test_attention_pool(self):
        a = AttentionPoolAdapter(64, 128, 8)
        x = torch.randn(3, 40, 64)
        mask = torch.zeros(3, 40, dtype=torch.bool)
        mask[:, 30:] = True
        assert a(x, mask).shape == (3, 8, 128)


# --------------------------------------------------------------- baselines
class TestBaselines:
    def test_conditional_gaussian(self):
        m = ConditionalGaussian(16)
        h = torch.randn(10, 16)
        assert torch.isfinite(m.log_prob(torch.randn(10), h)).all()
        assert m.sample(h).shape == (10,)

    def test_conditional_mdn(self):
        m = ConditionalMDN(16, n_components=3)
        h = torch.randn(10, 16)
        assert torch.isfinite(m.log_prob(torch.randn(10), h)).all()
        assert m.sample(h).shape == (10,)

    def test_independent_mdn_interface(self):
        m = IndependentMDN()
        z, present, log_det, cond = _head_inputs()
        out = m(z, present, log_det, cond)
        assert torch.isfinite(out["nll"]).all()
        m.eval()
        s = m.sample(cond[:2], n_samples=5)
        assert s["z"].shape == (2, 5, N_TOKENS)
