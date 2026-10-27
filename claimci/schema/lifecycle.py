"""A claim's stored status, as a small state machine.

A claim file stores only what a person or a verification run decided. Everything else a reader
sees (measured, stale, not holding, gated) is derived from the ledger on read and never written.

    proposed --run--> verified        a verification run reproduced the source (verify --promote)
    proposed --run--> unreproduced    it ran and did not reproduce it
    unreproduced --run--> verified    a later run did
    any --person--> refuted           a person records the opposite finding (qualifier NOT)
    refuted --person--> proposed      and can withdraw it
"""
from __future__ import annotations

STORED = ("proposed", "verified", "unreproduced", "refuted")

TRANSITIONS = {
    ("proposed", "verified"): "run",
    ("proposed", "unreproduced"): "run",
    ("unreproduced", "verified"): "run",
    ("proposed", "refuted"): "person",
    ("verified", "refuted"): "person",
    ("unreproduced", "refuted"): "person",
    ("refuted", "proposed"): "person",
}


class TransitionError(ValueError):
    pass


def check(current: str, new: str, by: str) -> None:
    """Raise unless `by` (run or person) may move a claim from `current` to `new`."""
    if current == new:
        return
    who = TRANSITIONS.get((current, new))
    if who is None:
        raise TransitionError(f"a claim cannot go from {current} to {new}")
    if who != by:
        raise TransitionError(f"{current} -> {new} is decided by a {who}, not a {by}")
