"""Inference operations on a loaded model: the only way experiments touch a model.

Each takes a handle (claimci/models.py) and formatted text (claimci/formats.py). Hooks go through
registered(), so they are removed on exit and the hook-leak canary applies. Training operations
(fine-tuning) belong here too when they arrive. A backend that computes differently (another
engine for generation, TransformerLens, nnsight) replaces these per operation.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .formats import with_bos


def _batches(h, texts: list[str]):
    bs = int(h.opts.get("batch", 16))
    for s in range(0, len(texts), bs):
        enc = h.tokenizer(texts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        yield {k: v.to(h.device) for k, v in enc.items()}


def last_logprobs(h, prompts: list[str], hooks: Iterable = (), keep: Callable | None = None):
    """Log-softmax at the last position for each formatted prompt, float64 on CPU.

    `keep` reduces [B, V] log-probs to something small per batch (e.g. a score), so
    full-vocabulary rows are never accumulated."""
    from .backends import registered
    out = []
    with torch.no_grad(), registered(hooks):
        for enc in _batches(h, with_bos(h, prompts)):
            lp = F.log_softmax(h.model(**enc).logits[:, -1, :].to(torch.float64), dim=-1).cpu()
            out.append(keep(lp) if keep else lp)
    return torch.cat(out)


def block_inputs(h, prompts: list[str], layer: int, pos: int) -> torch.Tensor:
    """[N, D] residual stream entering decoder block `layer` at position `pos` (negative, from the end)."""
    from .backends import registered
    got = []

    def grab(module, args):
        got.append(args[0][:, pos, :].detach().to(torch.float64).cpu())

    with torch.no_grad(), registered([(h.layers[layer], grab)]):
        for enc in _batches(h, with_bos(h, prompts)):
            h.model(**enc)
    return torch.cat(got)


def block_outputs(h, texts: list[str], block: int, pos: int | None = -1,
                  start: list[int] | None = None) -> torch.Tensor:
    """[N, D] output of decoder block `block`: at position `pos` (negative, from the end), or, with
    pos=None, the mean over positions from start[i] (counted from the left of each unpadded text)."""
    from .backends import registered
    got, offsets = [], []

    def grab(module, args, out):
        x = out[0] if isinstance(out, tuple) else out
        got.append(x.detach())

    bs = int(h.opts.get("batch", 16))
    texts = with_bos(h, texts)
    with torch.no_grad(), registered([(h.layers[block], grab, "post")]):
        for s in range(0, len(texts), bs):
            enc = h.tokenizer(texts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(h.device)
            h.model(**enc)
            x = got.pop().to(torch.float64)
            if pos is not None:
                offsets.append(x[:, pos, :].cpu())
                continue
            lengths = enc.attention_mask.sum(1).tolist()
            T = x.shape[1]
            for i, L in enumerate(lengths):  # left padding: the text occupies the last L positions
                first = T - L + start[s + i]
                offsets.append(x[i, first:, :].mean(0, keepdim=True).cpu())
    return torch.cat(offsets)


def generate(h, prompts: list[str], hooks=(), n: int = 1, max_new_tokens: int = 200,
             temperature: float = 1.0, seed: int = 0) -> list[list[str]]:
    """n sampled continuations per formatted prompt (greedy when temperature is 0)."""
    from .backends import registered
    out, bs = [], max(1, int(h.opts.get("batch", 8)) // max(n, 1))
    torch.manual_seed(seed)
    with torch.no_grad(), registered(hooks):
        for s in range(0, len(prompts), bs):
            enc = h.tokenizer(with_bos(h, prompts[s:s + bs]), return_tensors="pt", padding=True,
                              add_special_tokens=False).to(h.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
            seqs = h.model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                    pad_token_id=h.tokenizer.pad_token_id, **kw)
            T = enc.input_ids.shape[1]
            texts = [h.tokenizer.decode(q[T:], skip_special_tokens=True).strip() for q in seqs]
            out += [texts[i * n:(i + 1) * n] for i in range(len(texts) // n)]
    return out


def resid_norm_medians(h, prompt: str, layers: list[int]) -> dict[int, float]:
    """Per layer: median over tokens 1: of the residual norm leaving that layer, for one prompt."""
    enc = h.tokenizer(with_bos(h, [prompt]), return_tensors="pt", add_special_tokens=False).to(h.device)
    with torch.no_grad():
        hs = h.model(**enc, output_hidden_states=True).hidden_states
    return {L: float(hs[L + 1][0, 1:].float().norm(dim=-1).median()) for L in layers}


def text_loss(h, texts: list[str], hooks: Iterable = (), max_tokens: int = 256) -> float:
    """Mean next-token cross-entropy on plain texts (no chat template), one text at a time."""
    from .backends import registered
    total, n = 0.0, 0
    with torch.no_grad(), registered(hooks):
        for t in with_bos(h, texts):
            ids = h.tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_tokens].to(h.device)
            logits = h.model(input_ids=ids).logits[0, :-1].float()
            total += float(F.cross_entropy(logits, ids[0, 1:], reduction="sum"))
            n += ids.shape[1] - 1
    return total / n


def token_id(h, text: str) -> int:
    ids = h.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens for {h.name}; expected one")
    return ids[0]
