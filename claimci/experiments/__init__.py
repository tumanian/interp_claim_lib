"""Local experiment library.

Every experiment is registered under a stable id with `@experiment`. Claims name the id, never a
Python path, and their code version is the experiment's behaviour hash (claimci/schema/codehash.py), so
code can move without touching stored results. Each parameter has a kind, given by its type hint
(`block: Layer`, `pack: Data`; anything else is a scalar), which is what lets machinery such as
`expand` move an experiment to another model without knowing it:

  layer      a layer or block index: moved to the same relative depth
  component  neuron or head indices: specific to one model, never moved
  tokenizer  token ids: specific to one tokenizer, never moved
  input      a binding from an upstream claim (e.g. a direction): recomputed, never copied
  data       which data pack, split or file
  scalar     anything else (coefficients, sample counts, seeds, judge settings)
"""

from __future__ import annotations

from typing import Any

from ..run.outcome import Outcome, Pack

__all__ = ["Outcome", "Pack", "require", "experiment", "intervention", "Layer", "Component", "Tokenizer", "Input", "Data"]

KINDS = {"layer", "component", "tokenizer", "input", "data", "scalar"}

# Parameter kinds, as type hints. The registry reads the names; at run time the runner hands an
# `Input` its loaded value (a direction tensor, not an artifact reference) and a `Data` its Pack.
Layer = int
Component = Any
Tokenizer = Any
Input = Any
Data = Any


def require(ok: bool, why: str) -> None:
    """Stop the experiment with an error result when a precondition of the recipe does not hold."""
    if not ok:
        raise ValueError(why)


def experiment(eid: str, params: dict | None = None):
    """Register a function as experiment `eid`. Found statically (claimci/schema/codehash.py); the
    attributes set here are for code that already has the function in hand."""
    bad = {k: v for k, v in (params or {}).items() if v not in KINDS}
    if bad:
        raise ValueError(f"experiment {eid}: unknown parameter kinds {bad}")

    def wrap(fn):
        fn.experiment_id, fn.param_kinds = eid, dict(params or {})
        return fn
    return wrap


def intervention(fn):
    """Tag an entrypoint as an intervention (needs `ctx` to rerun the model with hooks).

    The validator detects this decorator statically; claims with causal evidence
    (IMP, ISS, ICA) must use a tagged entrypoint.
    """
    fn.intervention = True
    return fn
