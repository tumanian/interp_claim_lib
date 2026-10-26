"""Validation: typing rules from SPEC.md and graph checks from RUNNER.md. No model loads."""
from __future__ import annotations

from .datasets import content_hash, materialize
from .spec import Repo, SpecError, canonical

STATUSES = {"proposed", "verified", "stale", "refuted", "unreproduced"}
OPS = {">=", ">", "<=", "<", "=="}


def _cycles(repo: Repo) -> list[str]:
    errors, state = [], {}

    def visit(cid: str, stack: list[str]) -> None:
        state[cid] = "open"
        for d in repo.claims[cid].depends_on:
            if d not in repo.claims:
                continue
            if state.get(d) == "open":
                cyc = stack[stack.index(d):] + [d]
                errors.append(f"claim {d}: cycle in depends_on: {' -> '.join(cyc)}")
            elif d not in state:
                visit(d, stack + [d])
        state[cid] = "done"

    for cid in sorted(repo.claims):
        if cid not in state:
            visit(cid, [cid])
    return errors


def _claim_errors(repo: Repo, cid: str) -> list[str]:
    c, ev, out = repo.claims[cid], repo.evidence, []
    err = out.append
    code = ev.get(c.evidence)
    if code is None or not code.leaf:
        return [f"claim {cid}: unknown or non-leaf evidence code {c.evidence!r}"]
    if c.status not in STATUSES:
        err(f"claim {cid}: unknown status {c.status!r}")
    if c.qualifier not in ("", "NOT"):
        err(f"claim {cid}: qualifier must be '' or NOT, got {c.qualifier!r}")
    if c.question not in repo.questions:
        err(f"claim {cid}: question {c.question!r} not found")
    src = c.provenance.get("source") or {}
    if c.evidence == "TAS" and not src.get("locator"):
        err(f"claim {cid}: TAS requires provenance.source.locator")
    if c.evidence == "NAS" and src.get("locator"):
        err(f"claim {cid}: NAS claim has a locator; use TAS")
    if code.rerunnable:
        for k in ("metric", "expect"):
            if not getattr(c, k):
                err(f"claim {cid}: rerunnable claim needs `{k}`")
        if not c.code.get("entrypoint"):
            err(f"claim {cid}: rerunnable claim needs code.entrypoint")
        if c.expect and c.expect.get("op") not in OPS:
            err(f"claim {cid}: expect.op must be one of {sorted(OPS)}")
    # Rule 2: causal codes require an intervention entrypoint; an observational
    # entrypoint cannot be relabeled causal by editing the evidence field.
    if code.causal and "intervention" not in c.tags:
        err(f"claim {cid}: evidence {c.evidence} requires an entrypoint tagged @intervention")
    if not code.causal and "intervention" in c.tags:
        err(f"claim {cid}: entrypoint is an intervention but evidence {c.evidence} is not causal")
    for d in c.depends_on:
        parent = repo.claims.get(d)
        if parent is None:
            err(f"claim {cid}: depends_on names unknown claim {d}")
            continue
        pcode = ev.get(parent.evidence)
        if pcode is None:
            continue
        # Rule 1: NAS, TAS, IEA cannot be dependencies.
        if not pcode.dependency:
            err(f"claim {cid}: depends_on {d} has evidence {parent.evidence}, which cannot be a dependency")
        # Rule 3: ICP cannot be a dependency of causal claims.
        elif code.causal and not pcode.supports_causal:
            err(f"claim {cid}: causal claim ({c.evidence}) depends_on {d} with evidence "
                f"{parent.evidence}, which cannot support causal claims")
    for name, ref in c.inputs.items():
        if "." not in ref:
            err(f"claim {cid}: input {name}={ref!r} must be <claim_id>.<export>")
            continue
        pid, export = ref.rsplit(".", 1)
        if pid not in c.depends_on:
            err(f"claim {cid}: input {name} reads from {pid}, which is not in depends_on")
        elif pid in repo.claims and export not in repo.claims[pid].exports:
            err(f"claim {cid}: input {name} reads {pid}.{export}, but {pid} exports "
                f"{repo.claims[pid].exports}")
        # Rule 4: exports are consumed on the same data version unless cross_version.
        elif (pid in repo.claims and not c.cross_version and c.dataset and repo.claims[pid].dataset
              and c.dataset.ident != repo.claims[pid].dataset.ident):
            err(f"claim {cid}: input {name} crosses data versions from {pid}; set cross_version: true")
    return out


def _data_errors(repo: Repo) -> list[str]:
    """Generate each explicitly declared dataset once and compare to its declared hash."""
    out, seen = [], {}
    for cid in sorted(repo.claims):
        c = repo.claims[cid]
        if "dataset" not in c.data or c.dataset is None:
            continue
        dv = c.dataset
        if not dv.hash:
            out.append(f"claim {cid}: data.hash is not declared")
            continue
        key = canonical({"gen": dv.gen, "params": dv.params})
        if key not in seen:
            try:
                seen[key] = content_hash(materialize(dv))
            except (KeyError, TypeError) as e:
                seen[key] = None
                out.append(f"claim {cid}: dataset {dv.gen} cannot be generated: {e}")
        if seen[key] and seen[key] != dv.hash:
            out.append(f"claim {cid}: data.hash mismatch: declared {dv.hash}, generated {seen[key]}")
    return out


def validate(repo: Repo) -> list[str]:
    errors = list(repo.errors) + _data_errors(repo)
    cyc = _cycles(repo)
    errors += cyc
    for cid in sorted(repo.claims):
        errors += _claim_errors(repo, cid)
    for q in repo.questions.values():
        for cid in q.claims:
            if cid not in repo.claims:
                errors.append(f"question {q.id}: lists unknown claim {cid}")
    if not cyc and not errors:
        for cid in repo.claims:
            try:
                repo.claim_key(cid)
            except SpecError as e:
                errors.append(str(e))
    return errors
