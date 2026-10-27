"""Validation, as small checks over a loaded repo. No model loads.

    fields     the shape of a claim's own fields
    status     stored statuses consistent with evidence and kind
    evidence   the evidence ontology's rules, for a claim and its dependencies
    source     what a finding may and must carry
    graph      cycles and references between claims, questions and exports
    data       declared datasets hash as declared
"""
from __future__ import annotations

from ..schema.claim import SpecError
from . import evidence, fields, finding, graph, status
from .data import data_errors
from .rules import broken, first_broken


def claim_errors(repo, cid: str) -> list[str]:
    c = repo.claims[cid]
    code = repo.evidence.get(c.evidence)
    prefix = f"claim {cid}: "
    if code is None or not code.leaf:
        return [f"{prefix}unknown or non-leaf evidence code {c.evidence!r}"]
    subject = dict(c=c, repo=repo, code=code, earned=repo.earned_code(cid), src=c.provenance.get("source") or {})
    out = broken(status.EVERY_CLAIM + fields.EVERY_CLAIM, prefix, **subject)
    if c.is_finding:
        return (out + broken(finding.RULES + status.FINDING + fields.FINDING, prefix, **subject)
                + finding.stated_tests(c) + broken(graph.QUESTION, prefix, **subject))
    out += broken(fields.RUN + evidence.CLAIM, prefix, **subject)
    out += graph.references(repo, c)
    for d in c.depends_on:
        parent = repo.claims.get(d)
        out += first_broken(evidence.DEPENDENCY, prefix, **subject, d=d, parent=parent,
                            pcode=repo.evidence.get(parent.evidence) if parent else None)
    return out


def _keys(repo) -> list[str]:
    out = []
    for cid, c in repo.claims.items():
        if not c.is_finding:
            try:
                repo.claim_key(cid)
            except SpecError as e:
                out.append(str(e))
    return out


def validate(repo, data: bool = True) -> list[str]:
    errors = list(repo.errors) + (data_errors(repo) if data else []) + graph.cycles(repo)
    for cid in sorted(repo.claims):
        errors += claim_errors(repo, cid)
    errors += graph.question_lists(repo)
    return errors or _keys(repo)
