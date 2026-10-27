"""Chat models on prompt datasets: loading, prompt formatting, batched passes with hooks.

Experiments on chat models get these through `ctx` (claimci.backends.Ctx). Hooks use
the same registered() context manager as everything else, so they are removed on exit
and the hook-leak canary applies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

# Stated by Arditi et al. (refusal_direction, pipeline/model_utils/gemma_model.py).
GEMMA_PAPER_TEMPLATE = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


@dataclass
class ChatHandle:
    model: Any
    tokenizer: Any
    name: str
    revision: str
    opts: dict = field(default_factory=dict)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def layers(self):
        return self.model.model.layers

    def attn(self, i: int):
        layer = self.layers[i]
        return layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn

    def mlp(self, i: int):
        return self.layers[i].mlp

    def down_proj(self, i: int):
        return self.layers[i].mlp.down_proj


def load_chat(model: str, revision: str, opts: dict, offline: bool = False) -> ChatHandle:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, opts.get("dtype", "float32"))
    device = opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype,
                                             local_files_only=offline).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=offline)
    tok.padding_side = "left"  # the last position is the next-token position for every row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return ChatHandle(m, tok, model, revision, dict(opts))


def format_prompt(h: ChatHandle, instruction: str, system: str | None = None, prefix: str = "") -> str:
    """The chat-formatted prompt, ready for the assistant's first token. `system` adds a system turn
    (folded into the user turn for templates without one); `prefix` starts the assistant's answer."""
    template = h.opts.get("template", "hf")
    if template == "gemma_paper":
        text = f"{system}\n\n{instruction}" if system else instruction
        return GEMMA_PAPER_TEMPLATE.format(instruction=text) + prefix
    kw = {"enable_thinking": False} if h.opts.get("thinking") is False else {}
    msgs = [{"role": "user", "content": instruction}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    try:
        out = h.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # no system role in this template: fold it into the user turn
        msgs = [{"role": "user", "content": f"{system}\n\n{instruction}"}]
        out = h.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return out + prefix


def _batches(h: ChatHandle, texts: list[str]):
    bs = int(h.opts.get("batch", 16))
    for s in range(0, len(texts), bs):
        enc = h.tokenizer(texts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        yield {k: v.to(h.device) for k, v in enc.items()}


def _bos(h: ChatHandle, texts: list[str]) -> list[str]:
    """Prepend BOS where the tokenizer has one and the text does not start with it."""
    b = h.tokenizer.bos_token
    return [t if not b or t.startswith(b) else b + t for t in texts]


def last_logprobs(h: ChatHandle, prompts: list[str], hooks: Iterable = (), keep: Callable | None = None):
    """Log-softmax at the last position for each formatted prompt, float64 on CPU.

    `keep` reduces [B, V] log-probs to something small per batch (e.g. a score), so
    full-vocabulary rows are never accumulated."""
    from .backends import registered
    out = []
    with torch.no_grad(), registered(hooks):
        for enc in _batches(h, _bos(h, prompts)):
            lp = F.log_softmax(h.model(**enc).logits[:, -1, :].to(torch.float64), dim=-1).cpu()
            out.append(keep(lp) if keep else lp)
    return torch.cat(out)


def block_inputs(h: ChatHandle, prompts: list[str], layer: int, pos: int) -> torch.Tensor:
    """[N, D] residual stream entering decoder block `layer` at position `pos` (negative, from the end)."""
    from .backends import registered
    got = []

    def grab(module, args):
        got.append(args[0][:, pos, :].detach().to(torch.float64).cpu())

    with torch.no_grad(), registered([(h.layers[layer], grab)]):
        for enc in _batches(h, _bos(h, prompts)):
            h.model(**enc)
    return torch.cat(got)


def block_outputs(h: ChatHandle, texts: list[str], block: int, pos: int | None = -1,
                  start: list[int] | None = None) -> torch.Tensor:
    """[N, D] output of decoder block `block`: at position `pos` (negative, from the end), or, with
    pos=None, the mean over positions from start[i] (counted from the left of each unpadded text)."""
    from .backends import registered
    got, offsets = [], []

    def grab(module, args, out):
        x = out[0] if isinstance(out, tuple) else out
        got.append(x.detach())

    bs = int(h.opts.get("batch", 16))
    texts = _bos(h, texts)
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


def generate(h: ChatHandle, prompts: list[str], hooks=(), n: int = 1, max_new_tokens: int = 200,
             temperature: float = 1.0, seed: int = 0) -> list[list[str]]:
    """n sampled continuations per formatted prompt (greedy when temperature is 0)."""
    from .backends import registered
    out, bs = [], max(1, int(h.opts.get("batch", 8)) // max(n, 1))
    torch.manual_seed(seed)
    with torch.no_grad(), registered(hooks):
        for s in range(0, len(prompts), bs):
            enc = h.tokenizer(_bos(h, prompts[s:s + bs]), return_tensors="pt", padding=True,
                              add_special_tokens=False).to(h.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
            seqs = h.model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                    pad_token_id=h.tokenizer.pad_token_id, **kw)
            T = enc.input_ids.shape[1]
            texts = [h.tokenizer.decode(q[T:], skip_special_tokens=True).strip() for q in seqs]
            out += [texts[i * n:(i + 1) * n] for i in range(len(texts) // n)]
    return out


def resid_norm_medians(h: ChatHandle, prompt: str, layers: list[int]) -> dict[int, float]:
    """Per layer: median over tokens 1: of the residual norm leaving that layer, for one prompt."""
    enc = h.tokenizer(_bos(h, [prompt]), return_tensors="pt", add_special_tokens=False).to(h.device)
    with torch.no_grad():
        hs = h.model(**enc, output_hidden_states=True).hidden_states
    return {L: float(hs[L + 1][0, 1:].float().norm(dim=-1).median()) for L in layers}


def text_loss(h: ChatHandle, texts: list[str], hooks: Iterable = (), max_tokens: int = 256) -> float:
    """Mean next-token cross-entropy on plain texts (no chat template), one text at a time."""
    from .backends import registered
    total, n = 0.0, 0
    with torch.no_grad(), registered(hooks):
        for t in _bos(h, texts):
            ids = h.tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_tokens].to(h.device)
            logits = h.model(input_ids=ids).logits[0, :-1].float()
            total += float(F.cross_entropy(logits, ids[0, 1:], reduction="sum"))
            n += ids.shape[1] - 1
    return total / n


def token_id(h: ChatHandle, text: str) -> int:
    ids = h.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens for {h.name}; expected one")
    return ids[0]


@dataclass
class PromptActs:
    """The shared 'pass' for a prompt dataset: the data plus a canary probe.

    Chat experiments run their own passes through ctx; nothing large is cached."""
    data: dict
    probe: torch.Tensor
    meta: dict = field(default_factory=dict)


def canary_prompts(data) -> list[str] | None:
    """The first four prompts found in the dataset: strings, or items with a `question`."""
    if isinstance(data, list) and data:
        if isinstance(data[0], str):
            return data[:4]
        if isinstance(data[0], dict) and isinstance(data[0].get("question"), str):
            return [d["question"] for d in data[:4]]
    if isinstance(data, dict):
        for v in data.values():
            found = canary_prompts(v)
            if found:
                return found
    return None


def probe(h: ChatHandle, data: dict) -> torch.Tensor:
    found = canary_prompts(data)
    if not found:
        raise ValueError("prompt dataset has no prompts to probe")
    prompts = [format_prompt(h, p) for p in found]
    return last_logprobs(h, prompts, keep=lambda lp: lp.topk(20, dim=-1).values)
