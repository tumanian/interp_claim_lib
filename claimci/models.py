"""Language models as the backend loads them: weights on a device, with their tokenizer.

A handle exposes the model's blocks by role (layers, attention, MLP, down projection) so
interventions can name components without knowing each architecture's module paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LMHandle:
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


ChatHandle = LMHandle  # the name used before formats and models were separated


def load_lm(model: str, revision: str, opts: dict, device: str | None = None, offline: bool = False) -> LMHandle:
    """Load weights in opts["dtype"] onto opts["device"], else `device`, else CUDA when available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, opts.get("dtype", "float32"))
    device = opts.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype,
                                             local_files_only=offline).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=offline)
    tok.padding_side = "left"  # the last position is the next-token position for every row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return LMHandle(m, tok, model, revision, dict(opts))
