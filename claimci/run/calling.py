"""How the runner calls an experiment, and what it gets back: (value, bindings, interval).

Experiments are written as `fn(ctx, **params) -> Outcome` (claimci/run/outcome.py). The induction
experiments still use the first style, which reads the shared pass: `fn(acts, ...)` returning a
value and bindings, or just a value for an intervention."""
from __future__ import annotations

import importlib
import inspect

from .outcome import Outcome, Pack


def resolve_entrypoint(ep: str):
    mod, _, fn = ep.partition(":")
    return getattr(importlib.import_module(mod), fn)


def call(fn, ctx, claim, inputs: dict, intervention: bool) -> tuple[float, dict, tuple | None]:
    params = list(inspect.signature(fn).parameters)
    if params[0] == "ctx":
        value, bindings, interval = call_outcome(fn, ctx, claim, inputs)
    elif intervention:
        out = fn(ctx.acts, ctx, **inputs, **claim.params)
        value, bindings, interval = (out[0], {**inputs, **out[1]}, None) if isinstance(out, tuple) else (out, dict(inputs), None)
    elif "ctx" in params:   # observational, but runs its own passes
        (value, bindings), interval = fn(ctx.acts, ctx=ctx, **inputs, **claim.params), None
    else:
        (value, bindings), interval = fn(ctx.acts, **inputs, **claim.params), None
    missing = set(claim.exports) - set(bindings)
    if missing:
        raise ValueError(f"experiment did not return exports {sorted(missing)}")
    return round(float(value), 6), bindings, tuple(interval) if interval else None


def call_outcome(fn, ctx, claim, inputs: dict) -> tuple[float, dict, tuple | None]:
    """An `Input` arrives loaded (an artifact reference becomes its tensor), a `Data` parameter that
    names a pack of the dataset as that Pack (one naming a split within a pack stays a name); tensors
    it exports are saved as artifacts and its samples as a JSON artifact under `answers`."""
    from ..schema.experiments import lock
    kinds = lock()[claim.experiment_id]["params"]
    load = lambda v: ctx.artifacts.load(v) if isinstance(v, str) and v.startswith("artifact:") else v
    names_a_pack = lambda k, v: kinds.get(k) == "data" and isinstance(v, str) and v in ctx.acts.data
    params = {k: Pack(ctx.acts.data[v]) if names_a_pack(k, v) else v for k, v in claim.params.items()}
    out = fn(ctx, **{k: load(v) for k, v in inputs.items()}, **params)
    if not isinstance(out, Outcome):
        raise TypeError(f"{claim.experiment_id} returned {type(out).__name__}, not an Outcome")
    saved = {k: ctx.artifacts.save(v) if hasattr(v, "shape") else v for k, v in out.exports.items()}
    bindings = {**inputs, **out.details, **saved}
    if out.samples is not None:
        bindings["answers"] = ctx.artifacts.save_json(out.samples)
    return out.value, bindings, out.interval
