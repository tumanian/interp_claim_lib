"""The contract between the runner and an experiment written as `fn(ctx, **params) -> Outcome`."""
from __future__ import annotations

from dataclasses import dataclass, field


class Pack:
    """A data pack, read by attribute (`pack.test`, `pack.items`) or by key. Not a dict, so a pack
    field named like a dict method (`items`, `keys`) reads as the field."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name):
        return self._data[name]


@dataclass
class Outcome:
    """What an experiment returns. `value` is the number the claim checks; `exports` are what
    dependents read (tensors are saved as artifacts); `details` go into the ledger row as they are;
    `samples` (judged answers and the like) are saved for the report card's drill-down; `interval`
    is a 95% interval around the value (claimci/run/stats.py), used by verdicts when present."""
    value: float
    interval: tuple | None = None
    exports: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    samples: list | None = None
