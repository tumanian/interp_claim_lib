"""Adapters wrap a ModelHandle for experiment code. v0 ships the raw HF adapter only."""
from __future__ import annotations

from .backends import ModelHandle


class HFAdapter:
    """Locates components in HF GPT-NeoX (Pythia) and GPT-2 models."""

    def __init__(self, handle: ModelHandle):
        self.handle = handle
        self.model = handle.model
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads

    def layer(self, i: int):
        if hasattr(self.model, "gpt_neox"):
            return self.model.gpt_neox.layers[i]
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h[i]
        raise NotImplementedError(f"hf adapter: unsupported model {type(self.model).__name__}")

    def attn_out_proj(self, i: int):
        """The attention output projection; its input is [N, T, H * d_head], head-major."""
        layer = self.layer(i)
        return layer.attention.dense if hasattr(layer, "attention") else layer.attn.c_proj


ADAPTERS = {"hf": HFAdapter}
