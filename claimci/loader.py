"""Loaders: claim and question files -> Repo, with data.inherit and code.version resolved."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import yaml

from .spec import Claim, DataVersion, Question, Repo, SpecError, load_evidence, sha


CLAIM_FIELDS = {f for f in Claim.__dataclass_fields__} - {
    "path", "dataset", "code_version", "code_version_kind", "tags"}


def load_claim(path: Path) -> Claim:
    d = yaml.safe_load(path.read_text()) or {}
    cid = d.get("id", path.stem)
    unknown = set(d) - CLAIM_FIELDS
    if unknown:
        raise SpecError(cid, f"unknown fields {sorted(unknown)} in {path.name}")
    missing = {"id", "question", "evidence", "status", "statement", "subject", "code", "data"} - set(d)
    if missing:
        raise SpecError(cid, f"missing fields {sorted(missing)} in {path.name}")
    d["qualifier"] = d.get("qualifier") or ""
    for k in ("depends_on", "exports"):
        d[k] = list(d.get(k) or [])
    d["inputs"] = dict(d.get("inputs") or {})
    d["provenance"] = dict(d.get("provenance") or {})
    return Claim(**d, path=path)


def resolve_code(claim: Claim) -> None:
    """Set code_version: git:<sha> as declared, or sha(source + params) for local entrypoints."""
    ep = claim.code.get("entrypoint")
    ver = str(claim.code.get("version", "local"))
    if ver.startswith("git:"):
        claim.code_version, claim.code_version_kind = ver, "git"
        return
    if not ep:
        return
    mod, _, fn = ep.partition(":")
    try:
        spec = importlib.util.find_spec(mod)
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.origin:
        raise SpecError(claim.id, f"entrypoint module {mod!r} not found")
    src = Path(spec.origin).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            decos = [ast.unparse(d) for d in node.decorator_list]
            seg = ast.get_source_segment(src, node)
            claim.tags = frozenset(d.rsplit(".", 1)[-1] for d in decos)
            claim.code_version = "src:" + sha({"decorators": decos, "source": seg, "params": claim.params})[:16]
            claim.code_version_kind = "local"
            return
    raise SpecError(claim.id, f"entrypoint function {fn!r} not found in {mod}")


def _resolve_data(claims: dict[str, Claim], errors: list[str]) -> None:
    def resolve(cid: str, seen: tuple) -> DataVersion | None:
        c = claims[cid]
        if c.dataset:
            return c.dataset
        parent = c.data.get("inherit")
        if parent:
            if parent in seen:
                errors.append(f"claim {cid}: data.inherit cycle: {' -> '.join(seen + (parent,))}")
                return None
            if parent not in claims:
                errors.append(f"claim {cid}: data.inherit names unknown claim {parent}")
                return None
            c.dataset = resolve(parent, seen + (parent,))
        elif "dataset" in c.data:
            ds = c.data["dataset"]
            c.dataset = DataVersion(ds["gen"], dict(ds.get("params") or {}), c.data.get("hash"))
        else:
            errors.append(f"claim {cid}: data needs `dataset` or `inherit`")
        return c.dataset

    for cid in claims:
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
        questions[d["id"]] = Question(d["id"], d["text"], d.get("status", "open"),
                                      list(d.get("claims") or []), d.get("parent"), p)
    _resolve_data(claims, errors)
    for c in claims.values():
        try:
            resolve_code(c)
        except SpecError as e:
            errors.append(str(e))
    return Repo(root, claims, questions, load_evidence(), errors)
