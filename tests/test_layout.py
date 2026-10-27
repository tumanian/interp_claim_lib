"""The package is grouped by concern and imports point one way (DESIGN.md section 15):

    schema <- store <- run <- experiments;  sources on top;  read beside them;  app, cli above all.

Checked statically on every relative import, including ones inside functions."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "claimci"

ALLOWED = {
    "schema": {"schema"},
    "store": {"schema", "store"},
    "run": {"schema", "store", "run", "claimci"},          # claimci: __version__, recorded with each result
    "experiments": {"schema", "store", "run", "experiments"},
    "sources": {"schema", "store", "run", "sources"},
    "read": {"schema", "store", "experiments", "read"},    # experiments: the judge tool reuses the judge
}
EXCEPTIONS = {("claimci.schema.validate", "store")}       # validate checks that data files hash as declared


def _group(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and (PKG / parts[1]).is_dir() else "claimci"


def _imports():
    for f in PKG.rglob("*.py"):
        rel = f.relative_to(PKG.parent).with_suffix("").parts
        is_pkg = rel[-1] == "__init__"
        mod = ".".join(rel[:-1] if is_pkg else rel)
        base = mod.split(".") if is_pkg else mod.split(".")[:-1]
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.ImportFrom) and n.level:
                b = base[:len(base) - (n.level - 1)]
                if n.module:
                    yield mod, ".".join(b + [n.module])
                else:
                    yield from ((mod, ".".join(b + [a.name])) for a in n.names)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
                assert not any(x == "claimci" or x.startswith("claimci.") for x in names), \
                    f"{mod}: import claimci relatively"


def test_imports_point_one_way():
    bad = []
    for mod, target in _imports():
        src, dst = _group(mod), _group(target)
        if src in ALLOWED and dst not in ALLOWED[src] and (mod, dst) not in EXCEPTIONS:
            bad.append(f"{mod} -> {target}")
    assert bad == []


def test_top_level_is_only_entry_points():
    assert sorted(p.name for p in PKG.glob("*.py")) == ["__init__.py", "cli.py"]
