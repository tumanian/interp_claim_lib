"""Refusal direction (Arditi et al. 2024, arXiv:2406.11717) on chat models.

The data is the paper's own prompt samples (data/refusal_direction/, replayed from its
pipeline). Refusal is read the way the paper selects and filters: the log-odds of the
refusal tokens at the first generated position (its Appendix B), so every metric here is
deterministic and needs no generation or judge. Rates carry a Wilson interval over prompts.
"""
from __future__ import annotations

import torch

from . import Data, Input, Layer, Outcome, Tokenizer, experiment, intervention
from ..run import operations as ops
from ..run.formats import format_prompt
from ..run.interventions import add_at_input, project_out, random_like
from ..run.readouts import refusal_logodds
from ..run.stats import wilson


def or_random(direction: torch.Tensor, random_seed: int | None) -> torch.Tensor:
    """The direction, or, for a control, a random direction of the same norm."""
    d = direction.to(torch.float64)
    return d if random_seed is None else random_like(d, random_seed)


def refusal(scores: torch.Tensor) -> Outcome:
    """The share of prompts the model refuses (refusal score > 0), with a Wilson interval."""
    k, n = int((scores > 0).sum()), len(scores)
    return Outcome(value=float((scores > 0).double().mean()), interval=wilson(k, n),
                   details={"mean_score": round(float(scores.mean()), 4)})


def published(pack) -> torch.Tensor:
    """A direction the authors released, from data/ (named by the pack's `file`)."""
    from ..store.datasets import DATA_DIR
    return torch.load(DATA_DIR / pack.file, weights_only=True, map_location="cpu").to(torch.float64).flatten()


@experiment("refusal.direction")
def refusal_direction(ctx, layer: Layer, pos: int, refusal_toks: Tokenizer, filter_train: bool = True,
                      reference: Data = None):
    """Difference in means of the residual stream entering block `layer` at position `pos`,
    harmful minus harmless (paper Eq. 2), over the training samples the model itself treats as
    harmful (score > 0) and harmless (score < 0) when filter_train is set (the paper's default).
    Value: cosine with the published direction if `reference` names one, else the norm."""
    prompts = ctx.data("prompts")
    harmful, harmless = prompts.harmful_train, prompts.harmless_train
    if filter_train:
        harmful = [p for p, s in zip(harmful, refusal_logodds(ctx, harmful, refusal_toks)) if s > 0]
        harmless = [p for p, s in zip(harmless, refusal_logodds(ctx, harmless, refusal_toks)) if s < 0]
    mean_input = lambda ps: ops.block_inputs(ctx.handle, [format_prompt(ctx.handle, p) for p in ps], layer, pos).mean(0)
    direction = mean_input(harmful) - mean_input(harmless)
    ctx.count(2)
    value = (float(torch.nn.functional.cosine_similarity(direction, published(reference), dim=0)) if reference
             else float(direction.norm()))
    return Outcome(value=value, exports={"direction": direction},
                   details={"layer": layer, "pos": pos, "n_harmful": len(harmful), "n_harmless": len(harmless),
                            "norm": round(float(direction.norm()), 4)})


@experiment("refusal.rate")
def refusal_rate(ctx, split: Data, refusal_toks: Tokenizer):
    """Share of `split` the model refuses, no intervention."""
    return refusal(refusal_logodds(ctx, ctx.data("prompts")[split], refusal_toks))


@experiment("refusal.ablate")
@intervention
def ablation_refusal_rate(ctx, direction: Input, split: Data, refusal_toks: Tokenizer, random_seed: int | None = None):
    """Share of `split` refused with the direction (or a random one of the same norm) ablated."""
    ablate = project_out(ctx.handle, or_random(direction, random_seed))
    return refusal(refusal_logodds(ctx, ctx.data("prompts")[split], refusal_toks, ablate))


@experiment("refusal.add")
@intervention
def addition_refusal_rate(ctx, direction: Input, layer: Layer, split: Data, refusal_toks: Tokenizer,
                          coeff: float = 1.0, random_seed: int | None = None):
    """Share of `split` refused with the direction (or a random one of the same norm) added at the
    input of block `layer`, every position (paper Eq. 3; coefficient 1 on the raw mean difference)."""
    add = add_at_input(ctx.handle, or_random(direction, random_seed), layer, coeff)
    return refusal(refusal_logodds(ctx, ctx.data("prompts")[split], refusal_toks, add))


@experiment("refusal.ablate_text_loss")
@intervention
def ablation_text_loss_delta(ctx, direction: Input, n_texts: int = 32):
    """Cross-entropy on natural text with the direction ablated, minus without: is the model still
    coherent? (The paper reports CE loss for this, App. G.)"""
    texts = ctx.data("text").texts[:n_texts]
    before = ops.text_loss(ctx.handle, texts)
    after = ops.text_loss(ctx.handle, texts, project_out(ctx.handle, or_random(direction, None)))
    ctx.count(2)
    return Outcome(value=after - before, details={"base_loss": round(before, 4), "ablated_loss": round(after, 4)})
