"""Which stored statuses are consistent with a claim's evidence and kind."""
from __future__ import annotations

from ..schema.lifecycle import STORED
from .rules import Rule

UNVERIFIED = ("proposed", "unreproduced")

EVERY_CLAIM = [
    Rule(f"unknown status {{c.status!r}}; a claim file stores one of {list(STORED)} (measured, stale and the like are derived on read)",
         ok=lambda c, **_: c.status in STORED),
    Rule("status refuted requires qualifier NOT",
         when=lambda c, **_: c.status == "refuted", ok=lambda c, **_: c.qualifier == "NOT"),
    Rule("{c.evidence} claims stay proposed until an experiment reproduces them, got status {c.status!r}",
         when=lambda code, **_: not code.rerunnable, ok=lambda c, **_: c.status in UNVERIFIED),
]

SOURCE = [
    Rule("a source claim is never verified itself, got status {c.status!r}",
         ok=lambda c, **_: c.status in UNVERIFIED),
]
