"""Linear representations of sentiment (Tigges et al. 2023, arXiv:2310.15154) on base language models.

The direction: the mean output of block `layer` at the adjective in "I thought this movie was
{adj}", positive adjectives minus negative ones (the paper's mean-difference method, one of several
it finds agree). Half the adjectives build it; the other half are held out to test it.

Across tasks: the same direction, unchanged, read at the last token of Stanford Sentiment Treebank
sentences. Causally: zero-shot SST classification (does " positive" or " negative" come next after
"Review: ... Sentiment:"), with the direction projected out of every residual write, everywhere or
only at commas (the summarization motif). Accuracy is balanced over the two labels, and the effect
is the share of above-chance accuracy lost. A random direction of the same norm is the control.
"""
from __future__ import annotations

import torch

from . import Data, Input, Layer, Outcome, experiment, intervention, require
from ..run import operations as ops
from ..run.interventions import project_out, project_out_at, random_like
from ..run.readouts import balanced, balanced_interval
from ..run.stats import wilson

PROMPT = "Review: {sentence}\nSentiment:"


def at_adjective(ctx, pack, adjectives, layer) -> torch.Tensor:
    """[N, D] block output at each adjective (the last token of the toy sentence)."""
    return ops.block_outputs(ctx.handle, [pack.template.format(adj=a) for a in adjectives], layer, -1)


def by_label(item) -> int:
    return item["label"]


@experiment("sentiment.direction")
def sentiment_direction(ctx, layer: Layer, pack: Data):
    """Positive-minus-negative mean direction from half the adjectives. Value: the share of the
    held-out adjectives it classifies correctly (sign of the projection, centred between the means)."""
    pos, neg = at_adjective(ctx, pack, pack.train_positive, layer), at_adjective(ctx, pack, pack.train_negative, layer)
    direction = pos.mean(0) - neg.mean(0)
    u = direction / direction.norm()
    midpoint = float((pos.mean(0) + neg.mean(0)) @ u) / 2
    test_pos = at_adjective(ctx, pack, pack.test_positive, layer) @ u
    test_neg = at_adjective(ctx, pack, pack.test_negative, layer) @ u
    right = int((test_pos > midpoint).sum() + (test_neg < midpoint).sum())
    n = len(test_pos) + len(test_neg)
    ctx.count(4)
    return Outcome(value=right / n, interval=wilson(right, n), exports={"direction": direction},
                   details={"layer": layer, "midpoint": round(midpoint, 4), "n_train": len(pos) + len(neg), "n_test": n})


@experiment("sentiment.reads_sst")
def direction_reads_sst(ctx, direction: Input, layer: Layer, pack: Data):
    """The adjective direction, unchanged, on real reviews: balanced accuracy of the sign of the
    projection at each SST sentence's last token, centred at the median (no labels used to fit)."""
    items = pack.sst
    u = direction.to(torch.float64) / direction.norm()
    proj = ops.block_outputs(ctx.handle, [it["sentence"] for it in items], layer, -1) @ u
    said_positive = (proj > proj.median()).double()
    correct = torch.tensor([float(s == it["label"]) for s, it in zip(said_positive.tolist(), items)])
    ctx.count(1)
    return Outcome(value=balanced(items, correct, by_label), interval=balanced_interval(items, by_label, correct, combine=lambda c: c),
                   details={"n": len(items), "layer": layer})


def zero_shot(ctx, items, hooks=()) -> torch.Tensor:
    """[N] 1 where the model prefers the right label word next, else 0."""
    h = ctx.handle
    pos, neg = ops.token_id(h, " positive"), ops.token_id(h, " negative")
    lp = ops.last_logprobs(h, [PROMPT.format(sentence=it["sentence"]) for it in items], hooks, keep=lambda x: x[:, [pos, neg]])
    says_positive = lp[:, 0] > lp[:, 1]
    ctx.count(1)
    return torch.tensor([float(bool(p) == bool(it["label"])) for p, it in zip(says_positive.tolist(), items)])


@experiment("sentiment.ablate_zero_shot")
@intervention
def ablate_zero_shot(ctx, direction: Input, pack: Data, at: str = "everywhere", random_seed: int | None = None):
    """Share of above-chance zero-shot SST accuracy lost when the direction (or a random one of the
    same norm) is projected out of every residual write: everywhere, or only at commas."""
    items = pack.sst
    d = direction.to(torch.float64)
    d = d if random_seed is None else random_like(d, random_seed)
    ablate = project_out(ctx.handle, d) if at == "everywhere" else project_out_at(ctx.handle, d, ops.token_id(ctx.handle, " ,"))
    before, after = zero_shot(ctx, items), zero_shot(ctx, items, ablate)
    acc_before, acc_after = balanced(items, before, by_label), balanced(items, after, by_label)
    require(acc_before > 0.55, f"zero-shot accuracy {acc_before:.3f} is too close to chance to measure a loss")
    lost = lambda b, a: (b - a) / (b - 0.5)
    return Outcome(value=lost(acc_before, acc_after),
                   interval=balanced_interval(items, by_label, before, after, combine=lost),
                   details={"accuracy_before": round(acc_before, 4), "accuracy_after": round(acc_after, 4),
                            "at": at, "random_seed": random_seed, "n": len(items)})
