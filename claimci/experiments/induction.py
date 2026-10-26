"""Induction-head experiments. Each takes Acts (and ctx for interventions) plus declared inputs."""
from __future__ import annotations

import torch

from . import intervention


def _argmax_head(scores: torch.Tensor, min_layer: int = 0) -> tuple[float, dict]:
    """scores [L, H] -> (max value, {layer, head}) over layers >= min_layer."""
    masked = scores.clone()
    masked[:min_layer] = float("-inf")
    flat = int(torch.argmax(masked))
    layer, head = divmod(flat, scores.shape[1])
    return float(scores[layer, head]), {"layer": layer, "head": head}


def prev_token_scores(acts) -> torch.Tensor:
    """[L, H]: mean attention from position i to i-1, over i >= 1 and all sequences."""
    a = acts.attn  # [L, N, H, T, T]
    t = a.shape[-1]
    i = torch.arange(1, t)
    return a[..., i, i - 1].mean(dim=(1, 3))


def induction_scores(acts) -> torch.Tensor:
    """[L, H]: mean attention from second-half position i to i - half + 1."""
    half = acts.meta["params"]["half"]
    i = torch.arange(half, 2 * half)
    return acts.attn[..., i, i - half + 1].mean(dim=(1, 3))


def prev_token_score(acts):
    return _argmax_head(prev_token_scores(acts))


def induction_head_score(acts, min_layer):
    return _argmax_head(induction_scores(acts), min_layer)


def second_half_loss(acts) -> torch.Tensor:
    """Mean loss on predictions of second-copy tokens after the first (loss[:, i] predicts i+1)."""
    half = acts.meta["params"]["half"]
    return acts.loss[:, half:].mean()


@intervention
def ablate_head_loss_delta(acts, ctx, layer, head, ablation="zero"):
    if ablation != "zero":
        raise ValueError(f"unsupported ablation {ablation!r}")
    d = ctx.adapter.d_head

    def zero_head(module, args):
        x = args[0].clone()
        x[..., head * d:(head + 1) * d] = 0
        return (x,) + tuple(args[1:])

    ablated = ctx.run([(ctx.adapter.attn_out_proj(layer), zero_head)])
    return float(second_half_loss(ablated) - second_half_loss(acts))
