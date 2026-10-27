"""Validation: typing rules from SPEC.md and graph checks from RUNNER.md. No model loads."""
from __future__ import annotations

from .datasets import content_hash, materialize
from .spec import OPS, Repo, SpecError, canonical

STATUSES = {"proposed", "verified", "measured", "stale", "refuted", "unreproduced"}
KINDS = {"source", "test", "measurement"}
HEDGES = {"shows", "suggests", "speculates", "argues"}


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
    if c.status == "refuted" and c.qualifier != "NOT":
        err(f"claim {cid}: status refuted requires qualifier NOT")
    if not code.rerunnable and c.status not in ("proposed", "unreproduced"):
        err(f"claim {cid}: {c.evidence} claims stay proposed until an experiment reproduces them, "
            f"got status {c.status!r}")
    if c.kind not in KINDS:
        err(f"claim {cid}: kind must be one of {sorted(KINDS)}, got {c.kind!r}")
    if c.is_source:
        return out + _source_errors(repo, c)
    if c.kind == "measurement" and c.expect is not None:
        err(f"claim {cid}: a measurement has no `expect`; with a threshold it is a test")
    if c.kind == "test" and c.expect is None:
        err(f"claim {cid}: a test needs `expect`; without a threshold it is a measurement")
    for cv in c.caveats:
        if not (isinstance(cv, dict) and isinstance(cv.get("binding"), str) and cv.get("op") in OPS
                and isinstance(cv.get("value"), (int, float)) and isinstance(cv.get("text"), str)):
            err(f"claim {cid}: each caveat needs binding, op (one of {sorted(OPS)}), value and text")
    if c.control_for is not None and (repo.claims.get(c.control_for) is None or c.kind != "test"):
        err(f"claim {cid}: control_for must be on a test and name an existing claim, got {c.control_for!r}")
    if c.about is not None:
        target = repo.claims.get(c.about)
        if target is None or not target.is_source:
            err(f"claim {cid}: about must name a source claim, got {c.about!r}")
    if c.question not in repo.questions:
        err(f"claim {cid}: question {c.question!r} not found")
    src = c.provenance.get("source") or {}
    if c.evidence == "TAS" and not src.get("locator"):
        err(f"claim {cid}: TAS requires provenance.source.locator")
    if c.evidence == "NAS" and src.get("locator"):
        err(f"claim {cid}: NAS claim has a locator; use TAS")
    if code.rerunnable:
        for k in (("metric", "expect") if c.kind == "test" else ("metric",)):
            if not getattr(c, k):
                err(f"claim {cid}: rerunnable claim needs `{k}`")
        if not c.code.get("entrypoint"):
            err(f"claim {cid}: rerunnable claim needs code.entrypoint")
        supplemented = (c.provenance.get("supplemented") or {}).get("expect")
        if (c.kind == "test" and c.status == "proposed" and (src or c.about) and src.get("stated_value") is None
                and not supplemented):
            err(f"claim {cid}: source states no value, so `expect` must be recorded under "
                f"provenance.supplemented.expect")
    if c.expect is not None and not (isinstance(c.expect, dict) and c.expect.get("op") in OPS
                                     and isinstance(c.expect.get("value"), (int, float))
                                     and not isinstance(c.expect.get("value"), bool)):
        err(f"claim {cid}: expect must be {{op: one of {sorted(OPS)}, value: number}}")
    if c.tolerance is not None and not isinstance(c.tolerance, (int, float)):
        err(f"claim {cid}: tolerance must be a number")
    # Rule 2: causal codes require an intervention entrypoint; an observational
    # entrypoint cannot be relabeled causal by editing the evidence field. For a
    # TAS/NAS claim with an attached experiment the code it would earn is checked.
    earned = repo.earned_code(cid)
    if c.code.get("entrypoint") and not code.rerunnable and earned is None:
        err(f"claim {cid}: {c.evidence} claim with an entrypoint needs provenance.source.stated_evidence")
    if earned is not None:
        if earned.causal and "intervention" not in c.tags:
            err(f"claim {cid}: evidence {earned.id} requires an entrypoint tagged @intervention")
        if not earned.causal and "intervention" in c.tags:
            err(f"claim {cid}: entrypoint is an intervention but evidence {earned.id} is not causal")
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
        elif earned is not None and earned.causal and not pcode.supports_causal:
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


def _source_errors(repo: Repo, c) -> list[str]:
    """A source claim records what an author asserts: no experiment, no threshold."""
    out = []
    if c.evidence not in ("TAS", "NAS"):
        out.append(f"claim {c.id}: a source claim has author-statement evidence (TAS or NAS), got {c.evidence}")
    if c.code.get("entrypoint") or c.expect is not None:
        out.append(f"claim {c.id}: a source claim has no code or expect; attach a test with `about:` instead")
    if c.status not in ("proposed", "unreproduced"):
        out.append(f"claim {c.id}: a source claim is never verified itself, got status {c.status!r}")
    if c.hedge is not None and c.hedge not in HEDGES:
        out.append(f"claim {c.id}: hedge must be one of {sorted(HEDGES)}, got {c.hedge!r}")
    if not (c.provenance.get("source") or {}).get("ref"):
        out.append(f"claim {c.id}: a source claim needs provenance.source.ref")
    if c.question is not None and c.question not in repo.questions:
        out.append(f"claim {c.id}: question {c.question!r} not found")
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
            except Exception as e:  # bad params surface from the generator in many types
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
            if repo.claims[cid].is_source:
                continue
            try:
                repo.claim_key(cid)
            except SpecError as e:
                errors.append(str(e))
    return errors
