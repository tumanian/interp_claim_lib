"""Refusal direction (Arditi et al. 2024, arXiv:2406.11717) on chat models.

The data is the paper's own prompt samples (data/refusal_direction/, replayed from its
pipeline). Refusal is read the way the paper selects and filters: the log-odds of the
refusal tokens at the first generated position (its Appendix B), so every metric here is
deterministic and needs no generation or judge.
"""
from __future__ import annotations

import torch

from . import experiment, intervention
from .. import operations as ops
from ..formats import format_prompt
from ..interventions import add_at_input, project_out, random_like
from ..readouts import refusal_logodds


def _prompts(acts, split: str) -> list[str]:
    return acts.data["prompts"][split]


def _load(ctx, ref: str, random_seed=None) -> torch.Tensor:
    """The direction a claim received, or (for a control) a random one of the same norm."""
    d = ctx.artifacts.load(ref).to(torch.float64)
    return d if random_seed is None else random_like(d, random_seed)


@experiment("refusal.direction", params={'layer': 'layer', 'pos': 'scalar', 'refusal_toks': 'tokenizer', 'filter_train': 'scalar', 'reference': 'data'})
def refusal_direction(acts, layer, pos, refusal_toks, ctx=None, filter_train=True, reference=None):
    """Difference in means of the residual stream entering block `layer` at position `pos`,
    harmful minus harmless (paper Eq. 2), over the training samples the model itself treats
    as harmful (score > 0) and harmless (score < 0) when filter_train is set (the paper's
    default). Value: cosine with the published direction if `reference` names one, else the norm."""
    harm, harmless = _prompts(acts, "harmful_train"), _prompts(acts, "harmless_train")
    if filter_train:
        harm = [p for p, s in zip(harm, refusal_logodds(ctx, harm, refusal_toks)) if s > 0]
        harmless = [p for p, s in zip(harmless, refusal_logodds(ctx, harmless, refusal_toks)) if s < 0]
    h = ctx.handle
    mu = lambda ps: ops.block_inputs(h, [format_prompt(h, p) for p in ps], layer, pos).mean(0)
    d = mu(harm) - mu(harmless)
    ctx.count(2)
    bindings = {"direction": ctx.artifacts.save(d), "layer": layer, "pos": pos,
                "n_harmful": len(harm), "n_harmless": len(harmless), "norm": round(float(d.norm()), 4)}
    if reference:
        ref = torch.load(chat_data_path(acts, reference), weights_only=True, map_location="cpu").to(torch.float64).flatten()
        return float(torch.nn.functional.cosine_similarity(d, ref, dim=0)), bindings
    return float(d.norm()), bindings


def chat_data_path(acts, name: str):
    from ..datasets import DATA_DIR
    return DATA_DIR / acts.data[name]["file"]


@experiment("refusal.rate", params={'split': 'data', 'refusal_toks': 'tokenizer'})
def refusal_rate(acts, split, refusal_toks, ctx=None):
    """Fraction of `split` with refusal score > 0, no intervention."""
    s = refusal_logodds(ctx, _prompts(acts, split), refusal_toks)
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.ablate", params={'direction': 'input', 'split': 'data', 'refusal_toks': 'tokenizer', 'random_seed': 'scalar'})
@intervention
def ablation_refusal_rate(acts, ctx, direction, split, refusal_toks, random_seed=None):
    """Refusal rate on `split` with the direction (or a random one of the same norm) ablated."""
    d = _load(ctx, direction, random_seed)
    s = refusal_logodds(ctx, _prompts(acts, split), refusal_toks, project_out(ctx.handle, d))
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.add", params={'direction': 'input', 'layer': 'layer', 'split': 'data', 'refusal_toks': 'tokenizer', 'coeff': 'scalar', 'random_seed': 'scalar'})
@intervention
def addition_refusal_rate(acts, ctx, direction, layer, split, refusal_toks, coeff=1.0, random_seed=None):
    """Refusal rate on `split` with the direction (or a random one of the same norm) added at
    the input of block `layer`, every position (paper Eq. 3; coefficient 1 on the raw mean difference)."""
    d = _load(ctx, direction, random_seed)
    s = refusal_logodds(ctx, _prompts(acts, split), refusal_toks, add_at_input(ctx.handle, d, layer, coeff))
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.ablate_text_loss", params={'direction': 'input', 'n_texts': 'scalar'})
@intervention
def ablation_text_loss_delta(acts, ctx, direction, n_texts=32):
    """Cross-entropy on natural text with the direction ablated, minus without: is the model
    still coherent? (The paper reports CE loss for this, App. G.)"""
    texts = acts.data["text"]["texts"][:n_texts]
    base = ops.text_loss(ctx.handle, texts)
    ablated = ops.text_loss(ctx.handle, texts, project_out(ctx.handle, _load(ctx, direction, None)))
    ctx.count(2)
    return ablated - base, {"base_loss": round(base, 4), "ablated_loss": round(ablated, 4)}
