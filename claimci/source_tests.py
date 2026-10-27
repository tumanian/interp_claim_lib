"""The stored shape of the tests a source offers for a finding (`stated.tests` on a source claim).

Defined here, not by the extraction prompt: extraction maps Claude's output onto this shape
(claimci/sources/extract.py), validate checks it, and the two can change independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

EVIDENCE = ("observation", "probe", "ablation_or_patching", "steering", "causal_abstraction",
            "derivation", "benchmark", "qualitative_examples", "none")


@dataclass
class StatedValue:
    name: str
    value: str          # exactly as the source writes it ("0.91", "20%", "< 0.1%")
    context: str = ""


@dataclass
class SourceTest:
    description: str    # the experiment, concretely enough to rerun
    locator: str = ""   # section / figure / table / appendix
    evidence: str = "none"
    models: list[str] = field(default_factory=list)
    data: str = ""
    metric: str = ""
    quote: str = ""     # verbatim from the source's body; checked at ingest
    stated_values: list[StatedValue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceTest":
        return cls(**{**d, "stated_values": [StatedValue(**v) for v in d.get("stated_values") or []]})


def errors(tests) -> list[str]:
    """Why a stored `stated.tests` list does not have the SourceTest shape (empty when it does)."""
    if not isinstance(tests, list):
        return ["stated.tests must be a list"]
    out = []
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            out.append(f"stated.tests[{i}] must be a mapping")
            continue
        unknown = set(t) - set(SourceTest.__dataclass_fields__)
        if unknown:
            out.append(f"stated.tests[{i}] has unknown fields {sorted(unknown)}")
        if not isinstance(t.get("description"), str) or not t["description"].strip():
            out.append(f"stated.tests[{i}] needs a description")
        if t.get("evidence", "none") not in EVIDENCE:
            out.append(f"stated.tests[{i}].evidence must be one of {list(EVIDENCE)}")
        for k in ("locator", "data", "metric", "quote"):
            if not isinstance(t.get(k, ""), str):
                out.append(f"stated.tests[{i}].{k} must be text")
        if not isinstance(t.get("models", []), list):
            out.append(f"stated.tests[{i}].models must be a list")
        for j, v in enumerate(t.get("stated_values") or []):
            if not (isinstance(v, dict) and isinstance(v.get("name"), str) and isinstance(v.get("value"), str)
                    and set(v) <= {"name", "value", "context"}):
                out.append(f"stated.tests[{i}].stated_values[{j}] needs text name and value (and optional context)")
    return out
