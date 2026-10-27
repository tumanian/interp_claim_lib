"""What a downstream agent calls: search the base, get one result, re-verify what is stale.

    claimci search "refusal direction" --json
    claimci get refusal.gemma2b.ablation_jbb --json
    claimci reverify --dry-run            # what is stale, in the order it would run
    claimci reverify --limit 10           # run it (on the GPU box when .claimci-remote.json exists)
"""
from __future__ import annotations

import re
from collections import Counter

from . import cards
from ..store.ledger import Ledger
from .report import computed_status, current_row
from ..schema.repo import Repo

STALE_ORDER = {"code changed": 0, "data changed": 0, "model changed": 0, "parameters or inputs changed": 0, "never run": 1}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def search(repo: Repo, ledger: Ledger, query: str, kind: str | None = None, model: str | None = None,
           limit: int = 10) -> list[dict]:
    """Claims ranked by how many query words appear in their id, statement, quote, source title and model."""
    q = Counter(_words(query))
    out = []
    for cid, c in repo.claims.items():
        if kind and c.kind != kind:
            continue
        src = c.provenance.get("source") or {}
        if model and model.lower() not in (c.subject.get("model", {}).get("hf", "") or "").lower():
            continue
        hay = Counter(_words(" ".join([cid.replace(".", " "), c.statement, c.quote or "", src.get("title") or "",
                                       src.get("ref") or "", (c.subject.get("model") or {}).get("hf", "") or ""])))
        score = sum(min(n, hay[w]) * (2 if w in _words(cid) else 1) for w, n in q.items())
        if score:
            out.append((score, cid))
    out.sort(key=lambda x: (-x[0], x[1]))
    res = []
    for score, cid in out[:limit]:
        c = repo.claims[cid]
        res.append({"id": cid, "kind": cards.KIND_LABELS[c.kind], "statement": " ".join(c.statement.split()),
                    "verdict": (cards.card(repo, ledger, cid)["support"] if c.is_source else cards.verdict(repo, ledger, cid)),
                    "score": score})
    return res


def get(repo: Repo, ledger: Ledger, cid: str, revision: str | None = None) -> dict:
    """One claim's current result: enough for an agent to decide whether to build on it."""
    c = repo.claims[cid]
    if c.is_source:
        d = cards.card(repo, ledger, cid)
        return {"id": cid, "kind": "finding", "statement": d["statement"], "support": d["support"],
                "caveats": d.get("caveats", []), "verifications": [{"id": a["id"], "verdict": a["verdict"]} for a in d["attached"]]}
    rev = revision or c.revision
    r = current_row(repo, ledger, cid, rev)
    stale = ledger.get(cid, repo.claim_key(cid, rev)) is None
    out = {"id": cid, "kind": cards.KIND_LABELS[c.kind], "model": c.model, "revision": rev,
           "statement": " ".join(c.statement.split()), "stale": stale,
           "stale_reason": ledger.stale_reason(repo, cid, rev) if stale else None,
           "status": computed_status(repo, r, ledger) if r else "not run", "verdict": cards.verdict(repo, ledger, cid),
           "value": r.value if r else None, "interval": r.interval if r else None, "threshold": cards._threshold(c), "margin": c.margin(r.value, r.interval) if r else None,
           "post_hoc": bool(c.provenance.get("post_hoc")), "caveats": cards.caveats(repo, ledger, cid),
           "controls": [{"id": k, "verdict": cards.verdict(repo, ledger, k)} for k, o in repo.claims.items() if o.control_for == cid],
           "bindings": {k: v for k, v in (r.bindings if r else {}).items() if not str(v).startswith("artifact:")},
           "evidence": cards.EVIDENCE_LABELS.get(c.evidence), "about": c.about}
    return out


def _seconds(ledger: Ledger, c) -> float:
    """Expected cost of one claim: the mean of past runs of the same entrypoint on the same model."""
    past = [r.cost.get("seconds", 0.0) for r in ledger.rows
            if r.model == c.model and r.cost.get("seconds") and r.claim.split(".")[-1] == c.id.split(".")[-1]]
    return sum(past) / len(past) if past else 600.0


def reverify_plan(repo: Repo, ledger: Ledger, skip_models: tuple = ()) -> list[dict]:
    """Every stale runnable claim at its subject revision, ordered: previously verified claims that
    went stale first (keeping known results current), then never-run ones; cheapest first within each."""
    rows = []
    for cid, c in repo.claims.items():
        if c.is_source or not repo.runnable(cid) or any(s in c.model for s in skip_models):
            continue
        if ledger.get(cid, repo.claim_key(cid, c.revision)) is not None:
            continue
        why = ledger.stale_reason(repo, cid, c.revision)
        rows.append({"id": cid, "model": c.model, "why": why, "est_seconds": round(_seconds(ledger, c), 1)})
    rows.sort(key=lambda r: (STALE_ORDER.get(r["why"], 1), r["est_seconds"], r["id"]))
    return rows
