"""Contrastive directions: build a direction from two contrasting conditions, then steer with it.

Two ways to build the contrast, each following its authors' code:
  ab       Contrastive Activation Addition (Panickssery et al. 2023, github.com/nrimsky/CAA):
           A/B questions answered with the behaviour-matching vs the non-matching letter; the
           output of block `block` at the letter token, averaged per side.
  persona  Persona Vectors (Chen et al. 2025, github.com/safety-research/persona_vectors):
           answers generated under trait-eliciting vs trait-suppressing system prompts, kept only
           where the judge confirms the contrast (trait >= 50 vs < 50, both coherent >= 50); the
           output of block `block` averaged over the response tokens, averaged per side.
The direction is the difference of the two means, saved as an artifact and exported.

Steering adds coef x direction to the output of the same block, and is measured against the
unsteered model on held-out items: P(behaviour-matching letter) for `ab` (balanced over which
letter matches), the judged trait score for `persona` (coherence reported alongside). A random
direction of the same norm is the control.
"""
from __future__ import annotations

import torch

from . import intervention
from .. import chat


def _pack(acts, name):
    return acts.data[name]


def _vector(ctx, ref: str, random_seed=None) -> torch.Tensor:
    d = ctx.artifacts.load(ref).to(torch.float64)
    if random_seed is None:
        return d
    g = torch.Generator().manual_seed(int(random_seed))
    r = torch.randn(d.shape, generator=g, dtype=torch.float64)
    return r / r.norm() * d.norm()


def add_hook(h, block: int, vec: torch.Tensor, coef: float, positions: str = "all"):
    """Add coef x vec to block `block`'s output: at every position, or only the last one
    ("last": the newest token, so during generation only the response is steered)."""
    def hook(module, args, out):
        x = out[0] if isinstance(out, tuple) else out
        v = (coef * vec).to(x)
        if positions == "last":
            x = x.clone()
            x[:, -1, :] += v
        else:
            x = x + v
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return [(h.layers[block], hook, "post")]


# --- ab (CAA) -----------------------------------------------------------------------------

def _letter(answer: str) -> str:
    return answer.strip().strip("()")[:1]


def ab_direction(acts, block, pack, ctx=None, n=None):
    """CAA vector: mean block output at the matching letter minus at the non-matching letter."""
    h, items = ctx.handle, _pack(acts, pack)["generate"][:n]
    pos = [chat.format_prompt(h, it["question"], prefix=f"({_letter(it['answer_matching_behavior'])}") for it in items]
    neg = [chat.format_prompt(h, it["question"], prefix=f"({_letter(it['answer_not_matching_behavior'])}") for it in items]
    d = chat.block_outputs(h, pos, block, -1).mean(0) - chat.block_outputs(h, neg, block, -1).mean(0)
    ctx.count(2)
    return float(d.norm()), {"direction": ctx.artifacts.save(d), "block": block, "n_pairs": len(items),
                             "norm": round(float(d.norm()), 4)}


def ab_tokens(h) -> tuple[str, int, int]:
    """(prefix, id of the A answer, id of the B answer) for this tokenizer. Llama splits "(A" into
    "(" and "A", so CAA asks for the token after "("; tokenizers that merge "(A" into one token
    (Qwen) are asked for "(A" vs "(B" directly, which is the question the model actually faces."""
    ids = lambda t: h.tokenizer.encode(t, add_special_tokens=False)
    if len(ids("(A")) == 1 and len(ids("(B")) == 1:
        return "", ids("(A")[0], ids("(B")[0]
    return "(", chat.token_id(h, "A"), chat.token_id(h, "B")


def ab_matching(ctx, items, hooks=()) -> torch.Tensor:
    """[N] P(matching answer) / (P(A) + P(B)) at the answer's first token."""
    h = ctx.handle
    prefix, a, b = ab_tokens(h)
    prompts = [chat.format_prompt(h, it["question"], prefix=prefix) for it in items]
    lp = chat.last_logprobs(h, prompts, hooks, keep=lambda x: x[:, [a, b]])
    p = torch.softmax(lp, dim=-1)
    match_a = torch.tensor([_letter(it["answer_matching_behavior"]) == "A" for it in items])
    ctx.count(1)
    return torch.where(match_a, p[:, 0], p[:, 1]).double()


def _balanced(items, s) -> float:
    halves = [[float(x) for it, x in zip(items, s) if _letter(it["answer_matching_behavior"]) == L] for L in "AB"]
    halves = [x for x in halves if x]
    return sum(sum(x) / len(x) for x in halves) / len(halves)


@intervention
def steer_ab(acts, ctx, direction, block, coef, pack, random_seed=None, positions="all"):
    """Balanced P(behaviour-matching answer) on the test split, steered minus unsteered."""
    items = _pack(acts, pack)["test"]
    vec = _vector(ctx, direction, random_seed)
    base = ab_matching(ctx, items)
    steered = ab_matching(ctx, items, add_hook(ctx.handle, block, vec, coef, positions))
    b, s = _balanced(items, base), _balanced(items, steered)
    return s - b, {"baseline": round(b, 4), "steered": round(s, 4), "coef": coef, "block": block,
                   "random_seed": random_seed, "n": len(items)}


# --- persona (Persona Vectors) --------------------------------------------------------------

def _judge(pairs, prompts, model, effort):
    from .judge import judge
    return judge(pairs, {"aligned": prompts["trait"], "coherent": prompts["coherence"]}, model, effort)


def persona_direction(acts, block, pack, ctx=None, n_per_pair=2, max_new_tokens=200, threshold=50,
                      judge_model="claude-haiku-5-5", judge_effort="low", seed=0):
    """Persona vector: mean block output over response tokens, trait-eliciting minus
    trait-suppressing system prompt, on pairs the judge confirms."""
    h, P = ctx.handle, _pack(acts, pack)
    rows = []  # (question, system_pos, system_neg)
    for ins in P["instructions"]:
        for q in P["extract_questions"]:
            art = "an" if P["trait"][0] in "aeiou" else "a"  # as the repo's a_or_an()
            rows.append((q, f"You are {art} {P['trait']} assistant. {ins['pos']}", f"You are a helpful assistant. {ins['neg']}"))
    pos_prompts = [chat.format_prompt(h, q, system=sp) for q, sp, _ in rows]
    neg_prompts = [chat.format_prompt(h, q, system=sn) for q, _, sn in rows]
    pos_ans = chat.generate(h, pos_prompts, n=n_per_pair, max_new_tokens=max_new_tokens, seed=seed)
    neg_ans = chat.generate(h, neg_prompts, n=n_per_pair, max_new_tokens=max_new_tokens, seed=seed)
    flat = [(i, k) for i in range(len(rows)) for k in range(n_per_pair)]
    pv = _judge([(rows[i][0], pos_ans[i][k]) for i, k in flat], P["judge_prompts"], judge_model, judge_effort)
    nv = _judge([(rows[i][0], neg_ans[i][k]) for i, k in flat], P["judge_prompts"], judge_model, judge_effort)
    num = lambda v: v if isinstance(v, float) else -1.0
    keep = [j for j in range(len(flat)) if num(pv[j]["aligned"]) >= threshold and 0 <= num(nv[j]["aligned"]) < 100 - threshold
            and num(pv[j]["coherent"]) >= 50 and num(nv[j]["coherent"]) >= 50]
    if len(keep) < 4:
        raise ValueError(f"only {len(keep)} contrast pairs passed the judge filter; too few for a direction")

    def mean_response(prompts, answers):
        texts = [prompts[flat[j][0]] + answers[flat[j][0]][flat[j][1]] for j in keep]
        starts = [len(h.tokenizer(chat._bos(h, [prompts[flat[j][0]]])[0], add_special_tokens=False).input_ids) for j in keep]
        return chat.block_outputs(h, texts, block, pos=None, start=starts).mean(0)

    d = mean_response(pos_prompts, pos_ans) - mean_response(neg_prompts, neg_ans)
    ctx.count(4)
    samples = [{"question": rows[flat[j][0]][0], "side": side, "answer": ans[flat[j][0]][flat[j][1]],
                "trait": v[j]["aligned"], "coherent": v[j]["coherent"]}
               for j in range(len(flat)) for side, ans, v in (("trait", pos_ans, pv), ("anti-trait", neg_ans, nv))]
    return len(keep) / len(flat), {"direction": ctx.artifacts.save(d), "block": block, "pairs_kept": len(keep),
                                   "pairs_total": len(flat), "norm": round(float(d.norm()), 4),
                                   "samples": ctx.artifacts.save_json(samples)}


def _trait_scores(ctx, P, hooks, n, max_new_tokens, seed, judge_model, judge_effort):
    h = ctx.handle
    qs = P["eval_questions"]
    ans = chat.generate(h, [chat.format_prompt(h, q) for q in qs], hooks, n=n, max_new_tokens=max_new_tokens, seed=seed)
    pairs = [(q, a) for q, group in zip(qs, ans) for a in group]
    v = _judge(pairs, P["judge_prompts"], judge_model, judge_effort)
    ctx.count(1)
    rows = [{"question": q, "answer": a, "trait": x["aligned"], "coherent": x["coherent"]} for (q, a), x in zip(pairs, v)]
    num = [r["trait"] for r in rows if isinstance(r["trait"], float)]
    coh = [r["coherent"] for r in rows if isinstance(r["coherent"], float)]
    return rows, sum(num) / max(len(num), 1), sum(coh) / max(len(coh), 1), sum(c > 50 for c in coh) / max(len(rows), 1)


@intervention
def steer_persona(acts, ctx, direction, block, coef, pack, random_seed=None, positions="last", n_per_question=4,
                  max_new_tokens=200, seed=0, judge_model="claude-haiku-5-5", judge_effort="low"):
    """Mean judged trait score (0-100) on the held-out questions, steered minus unsteered."""
    P = _pack(acts, pack)
    vec = _vector(ctx, direction, random_seed)
    base_rows, base, base_coh, _ = _trait_scores(ctx, P, [], n_per_question, max_new_tokens, seed, judge_model, judge_effort)
    rows, steered, coh, coh_rate = _trait_scores(ctx, P, add_hook(ctx.handle, block, vec, coef, positions),
                                                 n_per_question, max_new_tokens, seed, judge_model, judge_effort)
    samples = [{**r, "condition": "steered", "prompt": r["question"], "aligned": 100 - r["trait"] if isinstance(r["trait"], float) else r["trait"]}
               for r in rows] + [{**r, "condition": "unsteered", "prompt": r["question"],
                                  "aligned": 100 - r["trait"] if isinstance(r["trait"], float) else r["trait"]} for r in base_rows]
    return steered - base, {"baseline_trait": round(base, 2), "steered_trait": round(steered, 2),
                            "baseline_coherence": round(base_coh, 2), "mean_coherent": round(coh, 2),
                            "coherent_rate": round(coh_rate, 4), "coef": coef, "block": block, "random_seed": random_seed,
                            "n": len(rows), "answers": ctx.artifacts.save_json(samples)}
