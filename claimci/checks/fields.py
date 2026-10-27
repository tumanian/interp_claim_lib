"""The shape of a claim's own fields: enums, thresholds, caveats. Nothing outside the claim."""
from __future__ import annotations

from ..schema.claim import OPS
from .rules import Rule

KINDS = {"finding", "test", "measurement"}
HEDGES = {"shows", "suggests", "speculates", "argues"}
LEVELS = {"annotation", "mechanism"}


def number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def threshold(e) -> bool:
    return (isinstance(e, dict) and e.get("op") in OPS and number(e.get("value")) and set(e) <= {"op", "value", "tol"}
            and (e.get("tol") is None or number(e["tol"])))


def caveat(cv) -> bool:
    return (isinstance(cv, dict) and isinstance(cv.get("binding"), str) and cv.get("op") in OPS
            and isinstance(cv.get("value"), (int, float)) and isinstance(cv.get("text"), str))


EVERY_CLAIM = [
    Rule("qualifier must be '' or NOT, got {c.qualifier!r}", ok=lambda c, **_: c.qualifier in ("", "NOT")),
    Rule(f"kind must be one of {sorted(KINDS)}, got {{c.kind!r}}", ok=lambda c, **_: c.kind in KINDS),
]

FINDING = [
    Rule(f"hedge must be one of {sorted(HEDGES)}, got {{c.hedge!r}}", ok=lambda c, **_: c.hedge is None or c.hedge in HEDGES),
    Rule(f"level must be one of {sorted(LEVELS)}, got {{c.level!r}}", ok=lambda c, **_: c.level is None or c.level in LEVELS),
]

RUN = [   # tests and measurements
    Rule("a measurement has no `expect`; with a threshold it is a test",
         when=lambda c, **_: c.kind == "measurement", ok=lambda c, **_: c.expect is None),
    Rule("a test needs `expect`; without a threshold it is a measurement",
         when=lambda c, **_: c.kind == "test", ok=lambda c, **_: c.expect is not None),
    Rule(f"each caveat needs binding, op (one of {sorted(OPS)}), value and text",
         ok=lambda c, **_: all(caveat(cv) for cv in c.caveats)),
    Rule(f"expect must be {{{{op: one of {sorted(OPS)}, value: number, tol: number (optional)}}}}",
         when=lambda c, **_: c.expect is not None, ok=lambda c, **_: threshold(c.expect)),
    Rule("tolerance must be a number",
         when=lambda c, **_: c.tolerance is not None, ok=lambda c, **_: isinstance(c.tolerance, (int, float))),
]
