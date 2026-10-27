"""Experiment registry and behaviour hash, both found by parsing source (nothing is imported).

Claims name experiments by a stable id (`code.experiment: caa.steer`), registered where the
function lives with `@experiment("caa.steer", params={...})`. A claim's code version is the
behaviour hash of that function: its syntax tree without docstrings, decorators or comments, plus,
transitively, every function, class and constant inside the claimci package that it reaches by
name, in any module. File paths (including constants derived from `__file__`), import lines and
module layout are not part of it, so moving or regrouping code changes no claim key, while
changing what an experiment computes, including a helper in another module, does.

Limit: calls through objects (`ctx.handle.layers`) cannot be resolved statically. Experiments
reach model operations through the operations module by name, so those are followed.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]   # the claimci package
PKG = "claimci"


@dataclass
class Module:
    name: str                                       # dotted, e.g. claimci.experiments.direction
    defs: dict[str, ast.AST] = field(default_factory=dict)       # top-level functions and classes
    consts: dict[str, ast.AST] = field(default_factory=dict)     # top-level assignments
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # local name -> (module, attr)


def _modname(path: Path) -> str:
    rel = path.relative_to(PKG_DIR.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(mod: str, level: int, target: str | None, is_pkg: bool) -> str:
    base = mod.split(".")
    if not is_pkg:
        base = base[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([target] if target else []))


def _imports_of(node, modname: str, is_pkg: bool) -> dict:
    out = {}
    if isinstance(node, ast.Import):
        for a in node.names:
            out[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
    elif isinstance(node, ast.ImportFrom):
        src = _resolve_relative(modname, node.level, node.module, is_pkg) if node.level else (node.module or "")
        for a in node.names:
            out[a.asname or a.name] = (src, a.name)
    return out


@lru_cache(maxsize=None)
def _modules() -> dict[str, Module]:
    mods = {}
    for path in sorted(PKG_DIR.rglob("*.py")):
        name, is_pkg = _modname(path), path.name == "__init__.py"
        m = Module(name)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                m.defs[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        m.consts[t.id] = node.value
                    elif isinstance(t, ast.Tuple):     # A, B = 30, 50: each name gets its own value
                        vals = node.value.elts if isinstance(node.value, ast.Tuple) and len(node.value.elts) == len(t.elts) else None
                        for i, e in enumerate(t.elts):
                            if isinstance(e, ast.Name):
                                m.consts[e.id] = vals[i] if vals else node.value
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                m.imports.update(_imports_of(node, name, is_pkg))
        mods[name] = m
    return mods


def clear_cache() -> None:
    _modules.cache_clear()
    registry.cache_clear()


# --- registry ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Registered:
    id: str
    module: str
    function: str
    params: tuple                 # ((name, kind), ...): what each parameter means across models
    tags: frozenset

    @property
    def entrypoint(self) -> str:
        return f"{self.module}:{self.function}"


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=None)
def registry() -> dict[str, Registered]:
    out = {}
    for m in _modules().values():
        for name, node in m.defs.items():
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            eid, params, tags = None, {}, set()
            for d in node.decorator_list:
                fn = d.func if isinstance(d, ast.Call) else d
                dname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if dname == "experiment" and isinstance(d, ast.Call):
                    eid = _literal(d.args[0]) if d.args else None
                    for kw in d.keywords:
                        if kw.arg == "params":
                            params = _literal(kw.value) or {}
                elif dname:
                    tags.add(dname)
            if eid:
                if eid in out:
                    raise ValueError(f"experiment id {eid!r} registered twice ({out[eid].entrypoint}, {m.name}:{name})")
                out[eid] = Registered(eid, m.name, name, tuple(sorted(params.items())), frozenset(tags))
    return out


def by_entrypoint(entrypoint: str) -> Registered | None:
    return next((r for r in registry().values() if r.entrypoint == entrypoint), None)


# --- behaviour hash ------------------------------------------------------------------------

class _Strip(ast.NodeTransformer):
    """Drop what does not change behaviour: docstrings, decorators, local imports' layout."""

    def _body(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        self.generic_visit(node)
        return self._body(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.decorator_list = []
        self.generic_visit(node)
        return self._body(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_AnnAssign(self, node):
        node.annotation = ast.Constant(None)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        return None

    visit_ImportFrom = visit_Import


def _canon(node) -> str:
    import copy
    if any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node)):
        return "<location>"                        # a path derived from where the file sits, not behaviour
    tree = _Strip().visit(copy.deepcopy(node))
    return ast.dump(tree, annotate_fields=False, include_attributes=False) if tree is not None else ""


def _deps(node, mod: Module, is_pkg: bool) -> list[tuple[str, str]]:
    """(module, name) pairs inside the package that `node` reaches by name."""
    mods = _modules()
    local_imports = {}
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            local_imports.update(_imports_of(sub, mod.name, is_pkg))
    imports = {**mod.imports, **local_imports}
    out = []

    def resolve(name: str, attr: str | None):
        if attr is None and (name in mod.defs or name in mod.consts):
            out.append((mod.name, name))
            return
        if name not in imports:
            return
        src, imported = imports[name]
        if not src.startswith(PKG):
            return
        if imported is None:                       # `import claimci.x as y` style: y.attr
            if attr and src in mods:
                out.append((src, attr))
            return
        full = f"{src}.{imported}"
        if full in mods:                           # `from .. import chat`: chat.attr
            if attr:
                out.append((full, attr))
        elif src in mods:                          # `from .judge import judge`
            out.append((src, imported))

    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            resolve(sub.value.id, sub.attr)
        elif isinstance(sub, ast.Name):
            resolve(sub.id, None)
    return out


def behaviour_hash(module: str, function: str) -> str:
    mods = _modules()
    seen, parts, todo = set(), [], [(module, function)]
    while todo:
        key = todo.pop()
        if key in seen:
            continue
        seen.add(key)
        mname, name = key
        m = mods.get(mname)
        if m is None:
            continue
        node = m.defs.get(name) or m.consts.get(name)
        if node is None:
            continue
        parts.append(f"{name}={_canon(node)}")     # by name and content, never by file path
        is_pkg = (PKG_DIR.parent / Path(*mname.split("."))).is_dir()
        todo += _deps(node, m, is_pkg)
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()[:16]
