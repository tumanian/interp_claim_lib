"""The contract between the runner and an experiment written as `fn(ctx, **params) -> Outcome`."""
from __future__ import annotations

from dataclasses import dataclass, field


class Pack(dict):
    """A data pack, read by attribute: `pack.test`, `pack.trait`."""
    __getattr__ = dict.__getitem__


@dataclass
class Outcome:
    """What an experiment returns. `value` is the number the claim checks; `exports` are what
    dependents read (tensors are saved as artifacts); `details` go into the ledger row as they are;
    `samples` (judged answers and the like) are saved for the report card's drill-down."""
    value: float
    exports: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    samples: list | None = None
