"""Loaders: claim and question files -> Repo, with data.inherit and code.version resolved."""
from __future__ import annotations

from pathlib import Path

import yaml

from .claim import Claim, DataVersion, Question, SpecError, load_evidence
from .repo import Repo


CLAIM_FIELDS = {f for f in Claim.__dataclass_fields__} - {
    "path", "dataset", "code_version", "code_version_kind", "tags", "experiment_id", "entrypoint"}


def load_claim(path: Path) -> Claim:
    d = yaml.safe_load(path.read_text()) or {}
    cid = d.get("id", path.stem)
    unknown = set(d) - CLAIM_FIELDS
    if unknown:
        raise SpecError(cid, f"unknown fields {sorted(unknown)} in {path.name}")
    d["kind"] = d.get("kind") or infer_kind(d)
    required = {"id", "evidence", "status", "statement"}
    if d["kind"] != "finding":
        required |= {"question", "subject", "code", "data"}
    missing = required - set(d)
    if missing:
        raise SpecError(cid, f"missing fields {sorted(missing)} in {path.name}")
    d["qualifier"] = d.get("qualifier") or ""
    d.setdefault("question", None)
    for k, t in (("depends_on", list), ("exports", list), ("inputs", dict), ("provenance", dict),
                 ("subject", dict), ("code", dict), ("data", dict), ("stated", dict)):
        d[k] = d.get(k) or t()
        if not isinstance(d[k], t):
            raise SpecError(cid, f"`{k}` must be a {t.__name__} in {path.name}")
    model = d["subject"].get("model")
    if d["kind"] != "finding" and (not isinstance(model, dict) or not isinstance(model.get("hf"), str)):
        raise SpecError(cid, f"subject.model.hf (str) is required in {path.name}")
    return Claim(**d, path=path)


def infer_kind(d: dict) -> str:
    """Files written before `kind` existed: an author statement with no experiment is a
    source, a claim with `expect` is a test, anything else is a measurement."""
    code = d.get("code") or {}
    if d.get("evidence") in ("TAS", "NAS") and not (code.get("entrypoint") or code.get("experiment")):
        return "finding"
    return "test" if d.get("expect") is not None else "measurement"


def resolve_code(claim: Claim) -> None:
    """Resolve the claim's experiment through versions.yaml and set its code version.

    `code.experiment` names a registered id. A legacy `code.entrypoint` (a Python path) is accepted
    if it points at a registered function. The code version is the experiment's version (`v3`), or
    the pinned `git:<sha>` for external code.
    """
    from .experiments import by_entrypoint, lock
    eid, ep = claim.code.get("experiment"), claim.code.get("entrypoint")
    if not eid and not ep:
        return
    found = (eid, lock()[eid]) if eid in lock() else None if eid else by_entrypoint(ep)
    if found is None:
        raise SpecError(claim.id, f"unknown experiment {eid!r} (not in versions.yaml; run `claimci versions`)" if eid
                        else f"entrypoint {ep!r} is not a registered experiment (add @experiment to it)")
    eid, entry = found
    claim.experiment_id, claim.entrypoint, claim.tags = eid, entry["entrypoint"], frozenset(entry.get("tags") or ())
    ver = str(claim.code.get("version", "local"))
    if ver.startswith("git:"):
        claim.code_version, claim.code_version_kind = ver, "git"
    else:
        claim.code_version, claim.code_version_kind = f"v{entry['version']}", "local"


def _resolve_data(claims: dict[str, Claim], errors: list[str]) -> None:
    def resolve(cid: str, seen: tuple) -> DataVersion | None:
        c = claims[cid]
        if c.dataset:
            return c.dataset
        parent = c.data.get("inherit")
        if parent and ("hash" in c.data or "dataset" in c.data):
            errors.append(f"claim {cid}: data.inherit cannot be combined with dataset/hash")
        elif parent:
            if parent in seen:
                errors.append(f"claim {cid}: data.inherit cycle: {' -> '.join(seen + (parent,))}")
                return None
            if parent not in claims:
                errors.append(f"claim {cid}: data.inherit names unknown claim {parent}")
                return None
            c.dataset = resolve(parent, seen + (parent,))
        elif "dataset" in c.data:
            ds = c.data["dataset"]
            if not isinstance(ds, dict) or not isinstance(ds.get("gen"), str) \
                    or not isinstance(ds.get("params", {}), dict):
                errors.append(f"claim {cid}: data.dataset needs `gen` (str) and `params` (mapping)")
                return None
            c.dataset = DataVersion(ds["gen"], dict(ds.get("params") or {}), c.data.get("hash"))
        else:
            errors.append(f"claim {cid}: data needs `dataset` or `inherit`")
        return c.dataset

    for cid in claims:
        if not claims[cid].is_finding:
            resolve(cid, (cid,))


def load_repo(root: Path | str = ".") -> Repo:
    root = Path(root)
    errors: list[str] = []
    claims: dict[str, Claim] = {}
    for p in sorted((root / "claims").glob("*.yaml")):
        try:
            c = load_claim(p)
        except SpecError as e:
            errors.append(str(e))
            continue
        if c.id in claims:
            errors.append(f"claim {c.id}: duplicate id in {p.name}")
        claims[c.id] = c
    questions = {}
    for p in sorted((root / "questions").glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        if not isinstance(d, dict) or not isinstance(d.get("id"), str) or not isinstance(d.get("text"), str):
            errors.append(f"question file {p.name}: needs `id` and `text`")
            continue
        questions[d["id"]] = Question(d["id"], d["text"], d.get("status", "open"),
                                      list(d.get("claims") or []), d.get("parent"), p)
    _resolve_data(claims, errors)
    for c in claims.values():
        try:
            resolve_code(c)
        except SpecError as e:
            errors.append(str(e))
    return Repo(root, claims, questions, load_evidence(), errors)


def write_promotion(claim: Claim, reproduced: bool, earned: str, basis: str = "stated_value") -> str:
    """The runner's only writes to a claim file: evidence, status, reproduced_from, basis.

    basis is stated_value (the source's number reproduced within tolerance) or
    supplemented_expect (the source states no value; judged by a threshold we set).

    Line edits, so comments and layout survive. Commit the change separately.
    """
    import re
    from .lifecycle import check
    check(claim.status, "verified" if reproduced else "unreproduced", by="run")
    text = claim.path.read_text()
    if reproduced:
        text = re.sub(r"(?m)^evidence: \S+", f"evidence: {earned}", text)
        text = re.sub(r"(?m)^status: \S+", "status: verified", text)
        ref = claim.provenance["source"]["ref"]
        text = re.sub(r"(?m)^  (reproduced_from|basis): .*\n", "", text)
        text = re.sub(r"(?m)^provenance:\n", f"provenance:\n  reproduced_from: {ref}\n  basis: {basis}\n", text)
    else:
        text = re.sub(r"(?m)^status: \S+", "status: unreproduced", text)
    claim.path.write_text(text)
    return "verified" if reproduced else "unreproduced"
