"""A loaded repository of claims: the dependency graph and the claim key."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .claim import Claim, EvidenceCode, Question, SpecError, sha


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
        return self.evidence.get(stated) if c.entrypoint and stated else None

    def runnable(self, cid: str) -> bool:
        code = self.earned_code(cid)
        return bool(code and code.rerunnable and self.claims[cid].entrypoint)

    def ancestors(self, cid: str) -> set[str]:
        out, frontier = set(), {cid}
        while frontier:
            nxt = {d for f in frontier for d in self.claims[f].depends_on if d in self.claims} - out
            out |= nxt
            frontier = nxt
        return out

    def lineage(self, cid: str) -> dict:
        # a claim's code is its experiment plus its parameters, so a parameter edit reads as "code changed"
        return {c: [f"{self.claims[c].code_version}+p{sha(self.claims[c].params)[:8]}",
                    self.claims[c].dataset.ident if self.claims[c].dataset else None]
                for c in sorted(self.ancestors(cid) | {cid})}

    def claim_key(self, cid: str, revision: str | None = None, _seen: tuple = ()) -> str:
        """sha(code.version + params + data.hash), plus model/revision and upstream keys.

        The experiment is its stable id plus behaviour hash, so file layout is not part of the key.
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
            "experiment": c.experiment_id, "code": c.code_version, "params": c.params,
            "inputs": c.inputs, "data": c.dataset.ident if c.dataset else None,
            "model": c.model, "revision": revision or c.revision, "upstream": upstream,
            **({"model_opts": opts} if (opts := {k: v for k, v in c.subject["model"].items()
                                                 if k not in ("hf", "revision")}) else {}),
        })[:16]
