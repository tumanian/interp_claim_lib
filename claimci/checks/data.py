"""Declared data: each dataset a claim names is generated once and must hash as declared."""
from __future__ import annotations

from ..schema.claim import canonical
from ..store.datasets import content_hash, materialize


def _declared(repo):
    for cid in sorted(repo.claims):
        c = repo.claims[cid]
        if "dataset" in c.data and c.dataset is not None:
            yield cid, c.dataset


def _generate(dv) -> tuple[str | None, str | None]:
    """(content hash, None) or (None, why it could not be generated)."""
    try:
        return content_hash(materialize(dv)), None
    except Exception as e:  # bad params surface from the generator in many types
        return None, str(e)


def data_errors(repo) -> list[str]:
    out, generated = [], {}
    for cid, dv in _declared(repo):
        if not dv.hash:
            out.append(f"claim {cid}: data.hash is not declared")
            continue
        spec = canonical({"gen": dv.gen, "params": dv.params})
        first = spec not in generated
        if first:
            generated[spec] = _generate(dv)
        got, why = generated[spec]
        if why and first:
            out.append(f"claim {cid}: dataset {dv.gen} cannot be generated: {why}")
        if got and got != dv.hash:
            out.append(f"claim {cid}: data.hash mismatch: declared {dv.hash}, generated {got}")
    return out
