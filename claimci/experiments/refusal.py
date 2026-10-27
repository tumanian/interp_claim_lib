"""Refusal direction (Arditi et al. 2024, arXiv:2406.11717) on chat models.

The data is the paper's own prompt samples (data/refusal_direction/, replayed from its
pipeline). Refusal is read the way the paper selects and filters: the log-odds of the
refusal tokens at the first generated position (its Appendix B), so every metric here is
deterministic and needs no generation or judge.
"""
from __future__ import annotations

import torch

from . import experiment, intervention
from .. import chat


def _prompts(acts, split: str) -> list[str]:
    return acts.data["prompts"][split]


def refusal_scores(ctx, instructions: list[str], refusal_toks: list[int], hooks=()) -> torch.Tensor:
    """[N] log P(refusal tokens) - log P(anything else) at the first response position."""
    h = ctx.handle
    prompts = [chat.format_prompt(h, i) for i in instructions]
    toks = torch.tensor(refusal_toks)

    def score(lp):
        p = lp.exp()[:, toks].sum(-1)
        return torch.log(p + 1e-8) - torch.log(1 - p + 1e-8)

    ctx.count(len(prompts) / max(len(prompts), 1))
    return chat.last_logprobs(h, prompts, hooks, keep=score)


def _direction(ctx, ref: str, random_seed: int | None) -> torch.Tensor:
    d = ctx.artifacts.load(ref).to(torch.float64)
    if random_seed is None:
        return d
    g = torch.Generator().manual_seed(int(random_seed))
    r = torch.randn(d.shape, generator=g, dtype=torch.float64)
    return r / r.norm() * d.norm()  # same norm, random orientation


def ablation_hooks(h, d: torch.Tensor):
    """Remove the component along d from every residual stream write, at every layer
    (block inputs, attention and MLP outputs), as in the paper's directional ablation."""
    u = d / d.norm()

    def project_out(x):
        uu = u.to(x)
        return x - (x @ uu).unsqueeze(-1) * uu

    def pre(module, args):
        return (project_out(args[0]),) + tuple(args[1:])

    def post(module, args, out):
        return (project_out(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else project_out(out)

    hooks = []
    for i in range(h.n_layers):
        hooks += [(h.layers[i], pre), (h.attn(i), post, "post"), (h.mlp(i), post, "post")]
    return hooks


def addition_hook(h, d: torch.Tensor, layer: int, coeff: float):
    def add(module, args):
        return (args[0] + coeff * d.to(args[0]),) + tuple(args[1:])
    return [(h.layers[layer], add)]


@experiment("refusal.direction", params={'layer': 'layer', 'pos': 'scalar', 'refusal_toks': 'tokenizer', 'filter_train': 'scalar', 'reference': 'data'})
def refusal_direction(acts, layer, pos, refusal_toks, ctx=None, filter_train=True, reference=None):
    """Difference in means of the residual stream entering block `layer` at position `pos`,
    harmful minus harmless (paper Eq. 2), over the training samples the model itself treats
    as harmful (score > 0) and harmless (score < 0) when filter_train is set (the paper's
    default). Value: cosine with the published direction if `reference` names one, else the norm."""
    harm, harmless = _prompts(acts, "harmful_train"), _prompts(acts, "harmless_train")
    if filter_train:
        harm = [p for p, s in zip(harm, refusal_scores(ctx, harm, refusal_toks)) if s > 0]
        harmless = [p for p, s in zip(harmless, refusal_scores(ctx, harmless, refusal_toks)) if s < 0]
    h = ctx.handle
    mu = lambda ps: chat.block_inputs(h, [chat.format_prompt(h, p) for p in ps], layer, pos).mean(0)
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
    s = refusal_scores(ctx, _prompts(acts, split), refusal_toks)
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.ablate", params={'direction': 'input', 'split': 'data', 'refusal_toks': 'tokenizer', 'random_seed': 'scalar'})
@intervention
def ablation_refusal_rate(acts, ctx, direction, split, refusal_toks, random_seed=None):
    """Refusal rate on `split` with the direction (or a random one of the same norm) ablated."""
    d = _direction(ctx, direction, random_seed)
    s = refusal_scores(ctx, _prompts(acts, split), refusal_toks, ablation_hooks(ctx.handle, d))
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.add", params={'direction': 'input', 'layer': 'layer', 'split': 'data', 'refusal_toks': 'tokenizer', 'coeff': 'scalar', 'random_seed': 'scalar'})
@intervention
def addition_refusal_rate(acts, ctx, direction, layer, split, refusal_toks, coeff=1.0, random_seed=None):
    """Refusal rate on `split` with the direction (or a random one of the same norm) added at
    the input of block `layer`, every position (paper Eq. 3; coefficient 1 on the raw mean difference)."""
    d = _direction(ctx, direction, random_seed)
    s = refusal_scores(ctx, _prompts(acts, split), refusal_toks, addition_hook(ctx.handle, d, layer, coeff))
    return float((s > 0).double().mean()), {"mean_score": round(float(s.mean()), 4)}


@experiment("refusal.ablate_text_loss", params={'direction': 'input', 'n_texts': 'scalar'})
@intervention
def ablation_text_loss_delta(acts, ctx, direction, n_texts=32):
    """Cross-entropy on natural text with the direction ablated, minus without: is the model
    still coherent? (The paper reports CE loss for this, App. G.)"""
    texts = acts.data["text"]["texts"][:n_texts]
    base = chat.text_loss(ctx.handle, texts)
    ablated = chat.text_loss(ctx.handle, texts, ablation_hooks(ctx.handle, _direction(ctx, direction, None)))
    ctx.count(2)
    return ablated - base, {"base_loss": round(base, 4), "ablated_loss": round(ablated, 4)}
