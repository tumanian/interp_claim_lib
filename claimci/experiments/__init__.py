"""Local experiment library.

Every experiment is registered under a stable id with `@experiment`. Claims name the id, never a
Python path, and their code version is the experiment's version in versions.yaml (DESIGN 19), so
code can move or be rewritten without touching stored results unless what it computes changes.
Each parameter has a kind, given by its type hint (`block: Layer`, `pack: Data`; anything else is
a scalar), which is what lets machinery such as `expand` move an experiment to another model
without knowing it:

  layer      a layer or block index: moved to the same relative depth
  component  neuron or head indices: specific to one model, never moved
  tokenizer  token ids: specific to one tokenizer, never moved
  input      a binding from an upstream claim (e.g. a direction): recomputed, never copied
  data       which data pack, split or file
  scalar     anything else (coefficients, sample counts, seeds, judge settings)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import NewType

from ..run.outcome import Outcome, Pack

__all__ = ["Outcome", "Pack", "require", "experiment", "intervention", "registry",
           "Layer", "Component", "Tokenizer", "Input", "Data"]

KINDS = {"layer", "component", "tokenizer", "input", "data", "scalar"}

# Parameter kinds, as type hints. At run time the runner hands an `Input` its loaded value (a
# direction tensor, not an artifact reference) and a `Data` parameter its Pack.
Layer = NewType("Layer", int)
Component = NewType("Component", object)
Tokenizer = NewType("Tokenizer", object)
Input = NewType("Input", object)
Data = NewType("Data", object)
MARKERS = {"Layer": "layer", "Component": "component", "Tokenizer": "tokenizer", "Input": "input", "Data": "data"}


@dataclass(frozen=True)
class Registered:
    id: str
    module: str
    function: str
    params: dict
    tags: frozenset

    @property
    def entrypoint(self) -> str:
        return f"{self.module}:{self.function}"


REGISTRY: dict[str, Registered] = {}


def _kinds(fn) -> dict:
    """Parameter kinds from type hints; a parameter without a marker is a scalar."""
    out = {}
    for name, p in inspect.signature(fn).parameters.items():
        if name in ("ctx", "acts"):
            continue
        hint = p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", "")
        out[name] = MARKERS.get(hint, "scalar")
    return out


def experiment(eid: str, params: dict | None = None):
    """Register a function as experiment `eid`. Older experiments declare kinds in `params`; newer
    ones by type hint."""
    bad = {k: v for k, v in (params or {}).items() if v not in KINDS}
    if bad:
        raise ValueError(f"experiment {eid}: unknown parameter kinds {bad}")

    def wrap(fn):
        if eid in REGISTRY and REGISTRY[eid].entrypoint != f"{fn.__module__}:{fn.__name__}":
            raise ValueError(f"experiment id {eid!r} registered twice ({REGISTRY[eid].entrypoint}, "
                             f"{fn.__module__}:{fn.__name__})")
        tags = frozenset({"intervention"} if getattr(fn, "intervention", False) else ())
        REGISTRY[eid] = Registered(eid, fn.__module__, fn.__name__, dict(params) if params else _kinds(fn), tags)
        fn.experiment_id = eid
        return fn
    return wrap


def intervention(fn):
    """Tag an experiment as an intervention (it reruns the model with hooks). Put it under
    `@experiment`. Claims with causal evidence (IMP, ISS, ICA) must use a tagged experiment."""
    fn.intervention = True
    return fn


def registry() -> dict[str, Registered]:
    """Every registered experiment: imports each module of this package once."""
    for m in pkgutil.iter_modules(__path__):
        if m.name != "versions":
            importlib.import_module(f"{__name__}.{m.name}")
    return REGISTRY


def require(ok: bool, why: str) -> None:
    """Stop the experiment with an error result when a precondition of the recipe does not hold."""
    if not ok:
        raise ValueError(why)
