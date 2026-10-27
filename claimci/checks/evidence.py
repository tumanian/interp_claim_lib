"""The evidence ontology's rules (evidence.yaml): what a code requires of a claim and of the
claims it depends on. Rule numbers are the ones in SPEC.md."""
from __future__ import annotations

from .rules import Rule


def _supplemented_expect(c, src) -> bool:
    return src.get("stated_value") is not None or bool((c.provenance.get("supplemented") or {}).get("expect"))


CLAIM = [
    Rule("TAS requires provenance.source.locator",
         when=lambda c, **_: c.evidence == "TAS", ok=lambda src, **_: bool(src.get("locator"))),
    Rule("NAS claim has a locator; use TAS",
         when=lambda c, **_: c.evidence == "NAS", ok=lambda src, **_: not src.get("locator")),
    Rule("rerunnable claim needs `metric`",
         when=lambda code, **_: code.rerunnable, ok=lambda c, **_: bool(c.metric)),
    Rule("rerunnable claim needs `expect`",
         when=lambda c, code, **_: code.rerunnable and c.kind == "test", ok=lambda c, **_: bool(c.expect)),
    Rule("rerunnable claim needs code.experiment",
         when=lambda code, **_: code.rerunnable, ok=lambda c, **_: bool(c.entrypoint)),
    Rule("source states no value, so `expect` must be recorded under provenance.supplemented.expect",
         when=lambda c, code, src, **_: code.rerunnable and c.kind == "test" and c.status == "proposed"
         and bool(src or c.about),
         ok=lambda c, src, **_: _supplemented_expect(c, src)),
    # Rule 2: causal codes require an intervention; an observational experiment cannot be relabeled
    # causal by editing the evidence field. For a TAS/NAS claim with an experiment, the code it earns.
    Rule("{c.evidence} claim with an experiment needs provenance.source.stated_evidence",
         when=lambda c, code, **_: bool(c.entrypoint) and not code.rerunnable, ok=lambda earned, **_: earned is not None),
    Rule("evidence {earned.id} requires an entrypoint tagged @intervention",
         when=lambda earned, **_: earned is not None and earned.causal, ok=lambda c, **_: "intervention" in c.tags),
    Rule("entrypoint is an intervention but evidence {earned.id} is not causal",
         when=lambda earned, **_: earned is not None and not earned.causal, ok=lambda c, **_: "intervention" not in c.tags),
]

DEPENDENCY = [   # one claim -> one parent it depends on; the first broken rule is reported
    Rule("depends_on names unknown claim {d}", ok=lambda parent, **_: parent is not None),
    # Rule 1: NAS, TAS, IEA cannot be dependencies.
    Rule("depends_on {d} has evidence {parent.evidence}, which cannot be a dependency",
         when=lambda pcode, **_: pcode is not None, ok=lambda pcode, **_: pcode.dependency),
    # Rule 3: ICP cannot be a dependency of causal claims.
    Rule("causal claim ({c.evidence}) depends_on {d} with evidence {parent.evidence}, which cannot support causal claims",
         when=lambda pcode, earned, **_: pcode is not None and earned is not None and earned.causal,
         ok=lambda pcode, **_: pcode.supports_causal),
]
