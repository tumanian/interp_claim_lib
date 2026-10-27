"""Backend interface, CpuBackend, Acts, and the hook context manager."""
from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .datasets import Dataset

Hook = tuple  # (module, fn) forward pre-hook fn(module, args), or (module, fn, "post") fn(module, args, out)


@dataclass
class ModelHandle:
    model: Any
    name: str
    revision: str

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def n_heads(self) -> int:
        return self.model.config.num_attention_heads


@dataclass
class Acts:
    hidden: torch.Tensor   # [L+1, N, T, D] residual stream (embeddings + each layer)
    attn: torch.Tensor     # [L, N, H, T, T] attention patterns
    loss: torch.Tensor     # [N, T-1] per-token loss; loss[:, i] predicts token i+1
    lengths: torch.Tensor  # [N]
    meta: dict = field(default_factory=dict)

    def equal(self, other: "Acts") -> bool:
        return all(torch.equal(getattr(self, k), getattr(other, k))
                   for k in ("hidden", "attn", "loss", "lengths")) and self.meta == other.meta


def acts_bytes(n_layers: int, n: int, t: int, d: int, h: int) -> int:
    """Shared-cache size: residual stream plus attention patterns, float32."""
    return 4 * ((n_layers + 1) * n * t * d + n_layers * n * h * t * t)


@contextmanager
def registered(hooks: Iterable[Hook]):
    """Register forward pre-hooks; remove every one on exit, exception or not."""
    handles = []
    try:
        for module, fn, *kind in hooks:
            if kind == ["post"]:
                handles.append(module.register_forward_hook(fn))
            else:
                handles.append(module.register_forward_pre_hook(fn))
        yield handles
    finally:
        for h in handles:
            h.remove()


class Backend:
    name = "abstract"

    def load(self, model: str, revision: str) -> ModelHandle:
        raise NotImplementedError

    def run(self, handle: ModelHandle, dataset: Dataset, hooks: Iterable[Hook] = ()) -> Acts:
        raise NotImplementedError

    def unload(self, handle: ModelHandle) -> None:
        raise NotImplementedError


class CpuBackend(Backend):
    """HF transformers on CPU: eager attention, no_grad, float32, pinned threads."""
    name = "cpu"
    loss_chunk = 8  # sequences per unembedding chunk

    def __init__(self, threads: int | None = None, offline: bool = False):
        torch.set_num_threads(threads or os.cpu_count() or 1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        self.offline = offline

    def load(self, model: str, revision: str, opts: dict | None = None):
        if opts and opts.get("chat"):
            from .chat import load_chat
            return load_chat(model, revision, opts, self.offline)
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, attn_implementation="eager",
            dtype=torch.float32, local_files_only=self.offline)
        return self.wrap(m, model, revision)

    @staticmethod
    def wrap(model, name: str, revision: str) -> ModelHandle:
        model.eval()
        return ModelHandle(model, name, revision)

    def run(self, handle: ModelHandle, dataset: Dataset, hooks: Iterable[Hook] = ()) -> Acts:
        if dataset.is_prompts:
            from .chat import PromptActs, probe
            return PromptActs(dataset.tokens, probe(handle, dataset.tokens), dataset.meta)
        ids = torch.tensor(dataset.tokens, dtype=torch.long)
        n, t = ids.shape
        model = handle.model
        with torch.no_grad(), registered(hooks):
            # Run the base model and unembed in chunks: full [N, T, vocab] logits
            # would dominate peak memory.
            out = model.base_model(input_ids=ids, output_hidden_states=True, output_attentions=True)
            unembed, losses = model.get_output_embeddings(), []
            for s in range(0, n, self.loss_chunk):
                logits = unembed(out.last_hidden_state[s:s + self.loss_chunk, :-1]).float()
                losses.append(F.cross_entropy(logits.flatten(0, 1), ids[s:s + self.loss_chunk, 1:].flatten(),
                                              reduction="none").view(logits.shape[0], t - 1))
            loss = torch.cat(losses)
        return Acts(hidden=torch.stack(out.hidden_states), attn=torch.stack(out.attentions),
                    loss=loss, lengths=torch.full((n,), t), meta=dataset.meta)

    def probe(self, handle, dataset: Dataset) -> torch.Tensor:
        """A small, fixed readout for the hook-leak canary: loss on 4 sequences, or top
        log-probs on 4 prompts for a prompt dataset."""
        if dataset.is_prompts:
            from .chat import probe
            return probe(handle, dataset.tokens)
        return self.run(handle, dataset.head(4)).loss

    @staticmethod
    def probe_of(acts) -> torch.Tensor:
        return acts.probe if hasattr(acts, "probe") else acts.loss[:4]

    def unload(self, handle: ModelHandle) -> None:
        handle.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class Ctx:
    """What an interventional experiment gets: rerun the group's dataset with hooks."""
    backend: Backend
    handle: ModelHandle
    dataset: Dataset
    adapter: Any
    passes: int = 0
    artifacts: Any = None  # claimci.artifacts.Store, set by the runner

    def run(self, hooks: Iterable[Hook]) -> Acts:
        self.passes += 1
        return self.backend.run(self.handle, self.dataset, hooks)

    def count(self, n: float = 1.0) -> None:
        """Chat experiments run their own batched passes; they report them here for cost."""
        self.passes += n


BACKENDS = {"cpu": CpuBackend}
