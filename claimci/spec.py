"""Spec types: Claim, Question, EvidenceCode, DataVersion, Result, Repo."""
from __future__ import annotations

import hashlib
import json
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
    kind: str | None = None        # source | test | measurement (inferred when absent)
    about: str | None = None       # the source claim a test or measurement bears on
    hedge: str | None = None       # source claims: shows | suggests | speculates | argues
    quote: str | None = None       # source claims: verbatim span from the source
    stated: dict = field(default_factory=dict)  # values the source states (intervention, recipe, ...)
    path: Path | None = None
    # resolved at load time
    dataset: DataVersion | None = None
    code_version: str | None = None
    code_version_kind: str | None = None  # "git" | "local"
    tags: frozenset[str] = frozenset()

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
    def revision(self) -> str:
        return str(self.subject["model"].get("revision", "main"))

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
    status: str  # ok | error | blocked | hook_leak
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


OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b,
       "<": lambda a, b: a < b, "==": lambda a, b: a == b}


def margin(expect: dict | None, value: float | None) -> float | None:
    """Signed distance from the threshold; positive means the expectation holds with room."""
    if expect is None or value is None or expect.get("op") == "==":
        return None
    d = value - expect["value"]
    return round(d if expect["op"] in (">=", ">") else -d, 6)


def evaluate(expect: dict | None, value: float | None) -> bool | None:
    """holds, from the current expect and a stored value (changing expect never reruns)."""
    if expect is None or value is None:
        return None
    return bool(OPS[expect["op"]](value, expect["value"]))


@dataclass
class Repo:
    root: Path
    claims: dict[str, Claim]
    questions: dict[str, Question]
    evidence: dict[str, EvidenceCode]
    errors: list[str] = field(default_factory=list)

    def dependents(self, cid: str) -> set[str]:
        out, frontier = set(), {cid}
        while frontier:
            nxt = {c.id for c in self.claims.values() if set(c.depends_on) & frontier} - out
            out |= nxt
            frontier = nxt
        return out

    def earned_code(self, cid: str) -> "EvidenceCode | None":
        """The evidence code a rerun earns: the claim's own if rerunnable, else the
        source's stated_evidence (a proposed TAS/NAS claim with an attached experiment)."""
        c = self.claims[cid]
        code = self.evidence.get(c.evidence)
        if code is None:
            return None
        if code.rerunnable:
            return code
        stated = (c.provenance.get("source") or {}).get("stated_evidence")
        return self.evidence.get(stated) if c.code.get("entrypoint") and stated else None

    def runnable(self, cid: str) -> bool:
        code = self.earned_code(cid)
        return bool(code and code.rerunnable and self.claims[cid].code.get("entrypoint"))

    def ancestors(self, cid: str) -> set[str]:
        out, frontier = set(), {cid}
        while frontier:
            nxt = {d for f in frontier for d in self.claims[f].depends_on if d in self.claims} - out
            out |= nxt
            frontier = nxt
        return out

    def lineage(self, cid: str) -> dict:
        return {c: [self.claims[c].code_version, self.claims[c].dataset.ident if self.claims[c].dataset else None]
                for c in sorted(self.ancestors(cid) | {cid})}

    def claim_key(self, cid: str, revision: str | None = None, _seen: tuple = ()) -> str:
        """sha(code.version + params + data.hash), plus model/revision and upstream keys.

        expect, statement and provenance are excluded: they change interpretation,
        not computation. Upstream keys are included so editing a claim makes its
        dependents stale.
        """
        if cid in _seen:
            raise SpecError(cid, "cycle in depends_on: " + " -> ".join(_seen + (cid,)))
        c = self.claims[cid]
        upstream = {d: self.claim_key(d, revision, _seen + (cid,))
                    for d in sorted(c.depends_on) if d in self.claims}
        return sha({
            "entrypoint": c.code.get("entrypoint"), "code": c.code_version, "params": c.params,
            "inputs": c.inputs, "data": c.dataset.ident if c.dataset else None,
            "model": c.model, "revision": revision or c.revision, "upstream": upstream,
            **({"model_opts": opts} if (opts := {k: v for k, v in c.subject["model"].items()
                                                 if k not in ("hf", "revision")}) else {}),
        })[:16]
