from pathlib import Path

import pytest
import torch

from claimci.run.adapters import HFAdapter
from claimci.run.backends import Acts, CpuBackend, Ctx
from claimci.store.datasets import load_dataset
from claimci.experiments.induction import (ablate_head_loss_delta, induction_head_score,
                                           prev_token_score)
from claimci.schema.loader import load_repo
from tests.models import cached_revisions, real, synthetic

ROOT = Path(__file__).resolve().parents[1]
DS = load_dataset(load_repo(ROOT).claims["mech.induction_head"].dataset)
MIN_LAYER = load_repo(ROOT).claims["mech.induction_head"].params["min_layer"]


def fake_acts(L=3, H=4, half=5, n=2):
    t = 2 * half
    attn = torch.zeros(L, n, H, t, t)
    attn[..., 0, 0] = 1.0
    return Acts(hidden=torch.zeros(L + 1, n, t, 8), attn=attn, loss=torch.zeros(n, t - 1),
                lengths=torch.full((n,), t), meta={"params": {"half": half}})


def test_prev_token_score_on_constructed_pattern():
    a = fake_acts()
    t = a.attn.shape[-1]
    for i in range(1, t):
        a.attn[1, :, 2, i, i - 1] = 1.0
    assert prev_token_score(a) == (1.0, {"layer": 1, "head": 2})


def test_induction_score_on_constructed_pattern():
    a = fake_acts()
    half = 5
    for i in range(half, 2 * half):
        a.attn[0, :, 0, i, i - half + 1] = 1.0   # layer 0: excluded by min_layer
        a.attn[2, :, 3, i, i - half + 1] = 0.5
    assert induction_head_score(a, min_layer=1) == (0.5, {"layer": 2, "head": 3})
    assert induction_head_score(a, min_layer=0) == (1.0, {"layer": 0, "head": 0})


def run_all(handle):
    backend = CpuBackend()
    acts = backend.run(handle, DS)
    prev = prev_token_score(acts)
    ind = induction_head_score(acts, MIN_LAYER)
    ctx = Ctx(backend, handle, DS, HFAdapter(handle))
    delta = ablate_head_loss_delta(acts, ctx, **ind[1])
    return acts, prev, ind, delta, ctx


def test_attention_rows_are_distributions():
    acts = CpuBackend().run(synthetic(), DS)
    sums = acts.attn.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    # causal: no attention to future positions
    assert acts.attn.triu(1).abs().max() < 1e-6


def test_ablation_runs_and_restores_on_synthetic():
    acts, prev, ind, delta, ctx = run_all(synthetic())
    print(f"\nsynthetic: prev={prev} induction={ind} ablation delta={delta:.6f}")
    assert ctx.passes == 1
    assert torch.equal(CpuBackend().run(ctx.handle, DS).loss, acts.loss)


def needs(rev):
    return pytest.mark.skipif(rev not in cached_revisions(),
                              reason=f"EleutherAI/pythia-70m@{rev} not in HF cache (tests never download)")


@needs("main")
def test_trained_model_has_heads_and_ablation_hurts():
    acts, prev, ind, delta, _ = run_all(real("main"))
    print(f"\nmain: prev_token={prev} induction={ind} ablation loss delta={delta:.6f} nats")
    sums = acts.attn.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert prev[0] >= 0.5
    assert ind[0] >= 0.3
    assert delta > 0
    if delta < 0.3:
        print(f"NOTE: ablation delta {delta:.6f} is under the claim threshold 0.3; threshold not changed")


@needs("step0")
def test_random_init_negative_control():
    acts, prev, ind, delta, _ = run_all(real("step0"))
    print(f"\nstep0: prev_token={prev} induction={ind} ablation loss delta={delta:.6f}")
    assert prev[0] < 0.15
    assert ind[0] < 0.15
