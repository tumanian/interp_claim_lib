"""Spec types: Claim, Question, EvidenceCode, DataVersion, Result, Repo."""
from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EVIDENCE_PATH = Path(__file__).with_name("evidence.yaml")
INHERITED = ("rerunnable", "dependency", "causal", "supports_causal")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


class SpecError(Exception):
    def __init__(self, claim_id: str, msg: str):
        super().__init__(f"claim {claim_id}: {msg}")
        self.claim_id = claim_id


@dataclass
class EvidenceCode:
    id: str
    definition: str
    parent: str | None = None
    name: str | None = None
    rerunnable: bool = False
    dependency: bool = False
    causal: bool = False
    supports_causal: bool = False
    leaf: bool = True


def load_evidence(path: Path = EVIDENCE_PATH) -> dict[str, EvidenceCode]:
    rows = yaml.safe_load(path.read_text())
    raw = {r["id"]: r for r in rows}
    codes: dict[str, EvidenceCode] = {}

    def build(cid: str) -> EvidenceCode:
        if cid in codes:
            return codes[cid]
        r = raw[cid]
        parent = build(r["parent"]) if r.get("parent") else None
        props = {k: r.get(k, getattr(parent, k) if parent else False) for k in INHERITED}
        codes[cid] = EvidenceCode(cid, r["definition"], r.get("parent"), r.get("name"), **props)
        return codes[cid]

    for cid in raw:
        build(cid)
    for c in codes.values():
        if c.parent:
            codes[c.parent].leaf = False
    return codes


@dataclass(frozen=True)
class DataVersion:
    gen: str
    params: dict
    hash: str | None = None  # declared content hash

    @property
    def ident(self) -> str:
        """Content hash if declared, else a hash of the generator spec."""
        return self.hash or "spec:" + sha({"gen": self.gen, "params": self.params})[:16]

    @property
    def short(self) -> str:
        return f"{self.gen}@{(self.hash or self.ident)[:4]}"


@dataclass
class Claim:
    id: str
    question: str
    evidence: str
    status: str
    statement: str
    subject: dict
    code: dict
    data: dict
    metric: str | None = None
    expect: dict | None = None
    tolerance: float | None = None
    qualifier: str = ""
    depends_on: list[str] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    exports: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    cross_version: bool = False
    kind: str | None = None  # source | test | measurement (inferred when absent)
    about: str | None = None       # the source claim a test or measurement bears on
    hedge: str | None = None       # source claims: shows | suggests | speculates | argues
    quote: str | None = None       # source claims: verbatim span from the source
    stated: dict = field(default_factory=dict)  # values the source states (intervention, recipe, ...)
    control_for: str | None = None  # a control test: the claim whose effect it must not reproduce
    caveats: list = field(default_factory=list)  # [{binding, op, value, text}]: qualifies a verdict, never gates it
    path: Path | None = None
    # resolved at load time
    dataset: DataVersion | None = None
    code_version: str | None = None
    code_version_kind: str | None = None  # "git" | "local"
    tags: frozenset[str] = frozenset()
    experiment_id: str | None = None  # resolved at load from the registry; entrypoint likewise,
    entrypoint: str | None = None     # never stored in claim files

    @property
    def is_source(self) -> bool:
        return self.kind == "source"

    @property
    def params(self) -> dict:
        return self.code.get("params") or {}

    @property
    def model(self) -> str:
        return self.subject["model"]["hf"]

    @property
    def model_opts(self) -> dict:
        return {k: v for k, v in self.subject["model"].items() if k not in ("hf", "revision")}

    @property
    def revision(self) -> str:
        return str(self.subject["model"].get("revision", "main"))

    @property
    def threshold(self) -> Expect | None:
        return Expect.of(self.expect)

    def holds(self, value: float | None, interval=None) -> bool | None:
        """Whether a value meets this claim's current threshold (None for a measurement)."""
        return self.threshold.holds(value, interval) if self.threshold else None

    def margin(self, value: float | None, interval=None) -> float | None:
        return self.threshold.margin(value, interval) if self.threshold else None

    def reproduction(self, value: float) -> dict | None:
        """For a proposed claim: does the value reproduce the source? Against the source's stated
        value within `tolerance` when there is one, else against a supplemented threshold."""
        target = (self.provenance.get("source") or {}).get("stated_value")
        if target is not None and self.tolerance is not None:
            return {"basis": "stated_value", "target": target, "within_tolerance": abs(value - target) <= self.tolerance}
        if target is None and (self.provenance.get("supplemented") or {}).get("expect"):
            return {"basis": "supplemented_expect", "expect": self.expect, "within_tolerance": bool(self.holds(value))}
        return None

    def input_refs(self) -> dict[str, tuple[str, str]]:
        """inputs as {name: (claim_id, export_name)}."""
        return {k: tuple(v.rsplit(".", 1)) for k, v in self.inputs.items()}

@dataclass
class Question:
    id: str
    text: str
    status: str
    claims: list[str]
    parent: str | None = None
    path: Path | None = None


@dataclass
class Result:
    claim: str
    claim_key: str
    status: str  # ok | error | blocked | contaminated (a hook outlived its intervention)
    value: float | None = None
    holds: bool | None = None
    bindings: dict = field(default_factory=dict)
    model: str | None = None
    revision: str | None = None
    code_version: str | None = None
    data_hash: str | None = None
    dataset: str | None = None
    reason: str | None = None
    traceback_hash: str | None = None
    lineage: dict = field(default_factory=dict)  # {claim_id: [code_version, data_ident]} self + ancestors
    reproduction: dict | None = None
    cost: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    migrated_from: str | None = None  # a copy of an earlier row under a re-derived key (same computation)


OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


@dataclass(frozen=True)
class Expect:
    """A claim's threshold: `value >= 0.5`, or `value == 0.91` within `tol`.

    Verdicts are derived from it on read, never stored, so editing a threshold re-judges every
    stored value without a rerun. With an interval (lo, hi) around the value, the threshold must
    hold for the whole interval, and the margin is measured from its near edge."""
    op: str
    value: float
    tol: float | None = None

    @classmethod
    def of(cls, d: dict | None) -> "Expect | None":
        return None if d is None else cls(d["op"], d["value"], d.get("tol"))

    def _edge(self, x: float, interval) -> float:
        lo, hi = interval or (x, x)
        return lo if self.op in (">=", ">") else hi

    def holds(self, x: float | None, interval=None) -> bool | None:
        if x is None:
            return None
        if self.op == "==":
            return abs(x - self.value) <= (self.tol or 0)
        return bool(OPS[self.op](self._edge(x, interval), self.value))

    def margin(self, x: float | None, interval=None) -> float | None:
        """Signed room past the threshold: positive when it holds, negative by how much it misses."""
        if x is None:
            return None
        if self.op == "==":
            return round((self.tol or 0) - abs(x - self.value), 6)
        d = self._edge(x, interval) - self.value
        return round(d if self.op in (">=", ">") else -d, 6)
