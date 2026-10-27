"""A finding records what an author asserts: no experiment, no threshold, a reference."""
from __future__ import annotations

from ..schema.source_tests import errors as stated_test_errors
from .rules import Rule

AUTHOR_STATEMENT = ("TAS", "NAS")

RULES = [
    Rule("a finding has author-statement evidence (TAS or NAS), got {c.evidence}",
         ok=lambda c, **_: c.evidence in AUTHOR_STATEMENT),
    Rule("a finding has no code or expect; attach a test with `about:` instead",
         ok=lambda c, **_: not (c.code.get("entrypoint") or c.code.get("experiment") or c.expect is not None)),
    Rule("a finding needs provenance.source.ref", ok=lambda src, **_: bool(src.get("ref"))),
]


def stated_tests(c) -> list[str]:
    """`stated.tests` has the stored SourceTest shape (schema/source_tests.py)."""
    if "tests" not in (c.stated or {}):
        return []
    return [f"claim {c.id}: {e}" for e in stated_test_errors(c.stated["tests"])]
