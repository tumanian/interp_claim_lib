"""Interventions: hooks that change what a model computes, reusable across experiments.

Each returns a list of hooks for registered() (claimci/run/backends.py): (module, fn) for a forward
pre-hook on a block's input, (module, fn, "post") for a forward hook on its output. Components are
named by role through the model handle (layers, attention, MLP, down projection), not by
architecture-specific module paths.
"""
from __future__ import annotations

import torch


def add_vector(h, block: int, vec: torch.Tensor, coef: float, positions: str = "all"):
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


def add_at_input(h, d: torch.Tensor, layer: int, coeff: float):
    """Add coeff x d to the residual stream entering block `layer`, at every position."""
    def add(module, args):
        return (args[0] + coeff * d.to(args[0]),) + tuple(args[1:])
    return [(h.layers[layer], add)]


def project_out(h, d: torch.Tensor):
    """Remove the component along d from every residual stream write, at every layer
    (block inputs, attention and MLP outputs): directional ablation."""
    u = d / d.norm()

    def project(x):
        uu = u.to(x)
        return x - (x @ uu).unsqueeze(-1) * uu

    def pre(module, args):
        return (project(args[0]),) + tuple(args[1:])

    def post(module, args, out):
        return (project(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else project(out)

    hooks = []
    for i in range(h.n_layers):
        hooks += [(h.layers[i], pre), (h.attn(i), post, "post"), (h.mlp(i), post, "post")]
    return hooks


def project_out_at(h, d: torch.Tensor, token_id: int):
    """Directional ablation as in project_out, but only at positions holding `token_id` (e.g. every
    comma): the input ids of each forward pass are read at the embedding."""
    u = d / d.norm()
    mask = {}

    def ids(module, args):
        mask["at"] = (args[0] == token_id).unsqueeze(-1)

    def project(x):
        uu = u.to(x)
        return torch.where(mask["at"].to(x.device), x - (x @ uu).unsqueeze(-1) * uu, x)

    def pre(module, args):
        return (project(args[0]),) + tuple(args[1:])

    def post(module, args, out):
        return (project(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else project(out)

    hooks = [(h.model.get_input_embeddings(), ids)]
    for i in range(h.n_layers):
        hooks += [(h.layers[i], pre), (h.attn(i), post, "post"), (h.mlp(i), post, "post")]
    return hooks


def add_to_neurons(h, interventions, medians: dict[int, float], scale: float = 1.0):
    """For each {layer, neuron, dose}: add dose x medians[layer] / ||W_down[:, neuron]|| to that
    neuron at the input of the layer's down projection, every position (the posted recipe)."""
    hooks = []
    for iv in interventions:
        L, n, dose = int(iv["layer"]), int(iv["neuron"]), float(iv["dose"]) * scale
        w = h.down_proj(L).weight[:, n].float().norm()
        amount = dose * medians[L] / float(w)

        def add(module, args, n=n, amount=amount):
            x = args[0].clone()
            x[..., n] += amount
            return (x,) + tuple(args[1:])
        hooks.append((h.down_proj(L), add))
    return hooks


# --- controls: same size, no meaning ----------------------------------------------------------

def random_like(d: torch.Tensor, seed: int) -> torch.Tensor:
    """A random direction with the norm of d (seeded)."""
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(d.shape, generator=g, dtype=torch.float64)
    return r / r.norm() * d.norm()


def random_neurons_like(h, interventions, seed: int):
    """Same layers, doses and count; neurons drawn uniformly at random (never the original)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for iv in interventions:
        width = h.down_proj(int(iv["layer"])).weight.shape[1]
        n = int(iv["neuron"])
        while n == int(iv["neuron"]):
            n = int(torch.randint(width, (1,), generator=g))
        out.append({**iv, "neuron": n})
    return out
