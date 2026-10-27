"""A check is a table of rules, not a chain of ifs.

A rule says what must hold, when it applies, and what to say when it does not. `broken` runs a
table against one subject (a claim, a dependency edge, an input) given as keyword arguments;
`first_broken` stops at the first failure, for rules that only make sense once earlier ones hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Pred = Callable[..., bool]


def always(**_) -> bool:
    return True


@dataclass(frozen=True)
class Rule:
    message: str        # str.format'ed with the subject's keyword arguments
    ok: Pred            # called with the subject's keyword arguments
    when: Pred = always


def broken(rules: list[Rule], prefix: str, **subject) -> list[str]:
    return [prefix + r.message.format(**subject) for r in rules if r.when(**subject) and not r.ok(**subject)]


def first_broken(rules: list[Rule], prefix: str, **subject) -> list[str]:
    for r in rules:                     # later rules may assume earlier ones held, so stop at the first
        if r.when(**subject) and not r.ok(**subject):
            return [prefix + r.message.format(**subject)]
    return []
