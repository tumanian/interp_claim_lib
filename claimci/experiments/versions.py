"""Experiment versions (DESIGN 19): versions.yaml next to this file, committed with the code.

Each experiment has a version, its entrypoint and parameter kinds, and a fingerprint: a hash of
the code that computes it (its module, and every experiment or run-layer module it reaches,
transitively). Orchestration (runner, planner, executors), schema and store are not part of it: they
do not change what an experiment computes, and data identity is in the key already. Claim keys
use the version, never the fingerprint. When code changes, the fingerprint no longer matches and
`claimci validate` fails until someone decides, in the same change:

    claimci versions --keep ID ...   same computation (a refactor): the version stays
    claimci versions --bump ID ...   different computation: the version goes up, its claims go stale

A new experiment gets version 1. The evidence for a keep is an equivalence run (old and new code on
the same inputs); a canary rerun can automate it later.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

from ..schema import experiments as lockfile
from . import registry

PKG_ROOT = Path(__file__).resolve().parents[1]


def _file(module: str) -> Path | None:
    """The source file of a claimci module, or None if `module` names something else."""
    base = PKG_ROOT.parent / Path(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imports(module: str, path: Path) -> set[str]:
    """claimci modules `module` imports, at any depth in the file (relative or absolute)."""
    is_pkg = path.name == "__init__.py"
    base = module.split(".") if is_pkg else module.split(".")[:-1]
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                parent = ".".join(base[:len(base) - (n.level - 1)] + ([n.module] if n.module else []))
            else:
                parent = n.module or ""
            if parent.split(".")[0] != "claimci":
                continue
            for a in n.names:   # `from .. import operations` names a module; `from .x import f` does not
                out.add(f"{parent}.{a.name}" if _file(f"{parent}.{a.name}") else parent)
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.split(".")[0] == "claimci"}
    return out


ORCHESTRATION = {"claimci.experiments", "claimci.run.runner", "claimci.run.verify", "claimci.run.planner",
                 "claimci.run.executors", "claimci.run.outcome", "claimci.run.adapters", "claimci.run.calling",
                 "claimci.run.guards", "claimci.run.rows"}


def computes(module: str) -> bool:
    return module.startswith(("claimci.experiments.", "claimci.run.")) and module not in ORCHESTRATION


def fingerprint(module: str) -> str:
    """Hash of the module's file and every file of computing code it reaches."""
    seen, todo = {}, [module]
    while todo:
        m = todo.pop()
        if m in seen or not computes(m):
            continue
        path = _file(m)
        if path is None:
            continue
        seen[m] = hashlib.sha256(path.read_bytes()).hexdigest()
        todo += sorted(_imports(m, path))
    rel = {str(_file(m).relative_to(PKG_ROOT)): h for m, h in seen.items()}
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(rel.items())).encode()).hexdigest()[:16]


def current() -> dict:
    """What versions.yaml should say about the code as it is, versions aside."""
    return {r.id: {"entrypoint": r.entrypoint, "params": dict(sorted(r.params.items())),
                   "tags": sorted(r.tags), "fingerprint": fingerprint(r.module)}
            for r in sorted(registry().values(), key=lambda r: r.id)}


def drift() -> dict[str, str]:
    """{experiment id: what changed} for every experiment whose lock entry is out of date."""
    locked, now = lockfile.lock(), current()
    out = {}
    for eid, entry in now.items():
        old = locked.get(eid)
        if old is None:
            out[eid] = "new experiment"
        elif old.get("entrypoint") != entry["entrypoint"] or old.get("params") != entry["params"] \
                or sorted(old.get("tags") or []) != entry["tags"]:
            out[eid] = "entrypoint, parameters or tags changed"
        elif old.get("fingerprint") != entry["fingerprint"]:
            out[eid] = "code changed"
    for eid in set(locked) - set(now):
        out[eid] = "no longer registered"
    return out


def write(keep=(), bump=(), new_only=False) -> dict[str, str]:
    """Update versions.yaml: new experiments get version 1; `keep` ids keep their version with the
    new fingerprint; `bump` ids go up by one. Returns {id: what was done}."""
    locked, now, done = dict(lockfile.lock()), current(), {}
    for eid, entry in now.items():
        old = locked.get(eid)
        if old is None:
            locked[eid], done[eid] = {"version": 1, **entry}, "added at v1"
        elif not new_only and eid in bump:
            locked[eid], done[eid] = {**entry, "version": old["version"] + 1}, f"bumped to v{old['version'] + 1}"
        elif not new_only and eid in keep:
            locked[eid], done[eid] = {**entry, "version": old["version"]}, f"kept at v{old['version']}"
    for eid in set(locked) - set(now):
        if eid in keep or eid in bump:
            del locked[eid]
            done[eid] = "removed"
    lockfile.LOCK_PATH.write_text("# Experiment versions (DESIGN 19). Written by `claimci versions`; commit it with the code.\n"
                         + yaml.safe_dump(dict(sorted(locked.items())), sort_keys=False, width=120))
    return done
