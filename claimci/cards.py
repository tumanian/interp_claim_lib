"""Claim cards and source explanations: what an agent (or the web app) reads.

A card is a short, honest summary of one claim: what it measures, on what, against
which threshold and who set it, every result with its margin, and its controls. For a
source claim the card lists the tests and measurements attached to it, so "asserted by
the source" and "measured on this model" never look alike.
"""
from __future__ import annotations

from .ledger import Ledger
from .report import computed_status, current_row, gated_by, revision_order
from .spec import Repo, margin


def _threshold(c) -> dict | None:
    if c.expect is None:
        return None
    sup = (c.provenance.get("supplemented") or {}).get("expect")
    src = (c.provenance.get("source") or {}).get("stated_value")
    origin = ("stated by the source" if src is not None
              else f"supplemented: {sup.get('note', '') if isinstance(sup, dict) else sup}" if sup else "unrecorded")
    return {"op": c.expect["op"], "value": c.expect["value"], "origin": origin}


def _results(repo: Repo, ledger: Ledger, cid: str) -> list[dict]:
    c, out = repo.claims[cid], []
    revs = sorted({r.revision for r in ledger.history(cid)}, key=revision_order) or [c.revision]
    for rev in revs:
        r = current_row(repo, ledger, cid, rev)
        if r is None:
            continue
        out.append({"model": r.model, "revision": rev, "status": computed_status(repo, r, ledger),
                    "value": r.value, "margin": margin(c.expect, r.value), "gated_by": gated_by(repo, ledger, cid, rev),
                    "bindings": {k: v for k, v in r.bindings.items() if not str(v).startswith("artifact:")},
                    "reason": r.reason, "seconds": round(r.cost.get("seconds", 0.0), 1), "at": r.run.get("at")})
    return out


def verdict(repo: Repo, ledger: Ledger, cid: str) -> str:
    """One word for a test at its subject revision, folding in its controls."""
    c = repo.claims[cid]
    rows = [x for x in _results(repo, ledger, cid) if x["revision"] == c.revision]
    if not rows:
        return "not run"
    status = rows[-1]["status"]
    if c.kind != "test" or status != "verified":
        return {"not holding": "fails", "verified": "holds"}.get(status, status)
    controls = [k for k, o in repo.claims.items() if o.control_for == cid]
    bad = [k for k in controls if verdict(repo, ledger, k) != "holds"]
    return "holds, control failed" if bad else "holds"


def card(repo: Repo, ledger: Ledger, cid: str) -> dict:
    c = repo.claims[cid]
    src = c.provenance.get("source") or {}
    d = {"id": cid, "kind": c.kind, "evidence": c.evidence, "statement": " ".join(c.statement.split()),
         "question": c.question, "about": c.about, "status_in_file": c.status}
    if c.is_source:
        d.update(hedge=c.hedge, quote=c.quote, source={k: src.get(k) for k in ("kind", "ref", "locator", "title", "date", "scope") if src.get(k)},
                 machine_made=str(c.provenance.get("assigned_by", "")).startswith("agent:ingest"),
                 checks=c.provenance.get("checks"), stated=c.stated or None,
                 source_tests=[{**t, "check": (c.provenance.get("checks") or {}).get("tests", [{}] * 99)[i]}
                               for i, t in enumerate((c.stated or {}).get("tests", []))])
        attached = sorted(k for k, o in repo.claims.items() if o.about == cid and not o.control_for)
        d["attached"] = [{"id": k, "kind": repo.claims[k].kind, "evidence": repo.claims[k].evidence,
                          "statement": " ".join(repo.claims[k].statement.split()), "metric": repo.claims[k].metric,
                          "model": repo.claims[k].model, "threshold": _threshold(repo.claims[k]),
                          "verdict": verdict(repo, ledger, k), "results": _results(repo, ledger, k),
                          "controls": [{"id": j, "statement": " ".join(o.statement.split()),
                                        "threshold": _threshold(o), "verdict": verdict(repo, ledger, j),
                                        "results": _results(repo, ledger, j)}
                                       for j, o in sorted(repo.claims.items()) if o.control_for == k]}
                         for k in attached]
        tests = [a for a in d["attached"] if a["kind"] == "test"]
        d["support"] = ("untested" if not tests else
                        "tests attached, not run" if all(a["verdict"] == "not run" for a in tests) else
                        "contradicted by a test" if any(a["verdict"] == "fails" for a in tests) else
                        "not supported: a control reproduced the effect" if any(a["verdict"] == "holds, control failed" for a in tests) else
                        "supported" if all(a["verdict"] == "holds" for a in tests) else "partly tested")
        return d
    d.update(model=c.model, model_opts=c.model_opts, data=c.dataset.short if c.dataset else None,
             metric=c.metric, threshold=_threshold(c), control_for=c.control_for,
             supplemented=sorted((c.provenance.get("supplemented") or {})), stated=c.stated or None,
             results=_results(repo, ledger, cid), verdict=verdict(repo, ledger, cid),
             controls=[{"id": k, "verdict": verdict(repo, ledger, k), "results": _results(repo, ledger, k)}
                       for k, o in sorted(repo.claims.items()) if o.control_for == cid])
    return d


def explain(repo: Repo, ledger: Ledger, ref: str) -> list[dict]:
    """Cards for every source claim from one source (an arXiv id, a URL, or a claim id)."""
    if ref in repo.claims:
        return [card(repo, ledger, ref)]
    norm = ref.lower().replace("arxiv:", "").strip()
    ids = sorted(k for k, c in repo.claims.items() if c.is_source
                 and norm in str((c.provenance.get("source") or {}).get("ref", "")).lower())
    return [card(repo, ledger, k) for k in ids]


def render(d: dict) -> str:
    """Plain-text card, sized for an agent's context."""
    lines = [f"{d['id']}  [{d['kind']}, {d['evidence']}]" + (f"  about: {d['about']}" if d.get("about") else "")]
    lines.append(f"  {d['statement']}")
    if d["kind"] == "source":
        s = d["source"]
        lines.append(f"  source: {s.get('kind')} {s.get('ref')} {s.get('locator', '')}  hedge: {d['hedge']}"
                     + ("  [machine-made, unreviewed]" if d["machine_made"] else ""))
        if d.get("checks"):
            lines.append(f"  quote check: {'found in source' if d['checks']['quote_found'] else 'NOT FOUND in source'}")
        for t in d.get("source_tests", []):
            ok = t["check"].get("quote_found")
            vals = ", ".join(f"{v['name']} {v['value']}" for v in t["stated_values"][:4])
            lines.append(f"  source's test ({t['locator']}, {t['evidence']}): {t['description'][:140]}"
                         + (f" [{vals}]" if vals else "") + ("" if ok is None else f" quote {'found' if ok else 'NOT FOUND'}"))
        lines.append(f"  support: {d['support']}")
        for a in d["attached"]:
            r = a["results"][-1] if a["results"] else None
            val = f" value {r['value']:.4g}" if r and r["value"] is not None else ""
            lines.append(f"    - {a['id']} [{a['kind']}, {a['evidence']}]: {a['verdict']}{val}")
        return "\n".join(lines)
    lines.append(f"  measures: {d['metric']} on {d['model']} {d['model_opts'] or ''}, data {d['data']}")
    if d["threshold"]:
        t = d["threshold"]
        lines.append(f"  threshold: {t['op']} {t['value']} ({t['origin']})")
    if d["supplemented"]:
        lines.append(f"  our choices (supplemented): {', '.join(d['supplemented'])}")
    for r in d["results"]:
        m = f"  margin {r['margin']:+.4g}" if r["margin"] is not None else ""
        v = f"{r['value']:.4g}" if r["value"] is not None else "-"
        lines.append(f"  result {r['model']}@{r['revision']}: {v} {r['status']}{m}"
                     + (f"  gated by {', '.join(r['gated_by'])}" if r["gated_by"] else ""))
    for k in d["controls"]:
        lines.append(f"  control {k['id']}: {k['verdict']}")
    lines.append(f"  verdict: {d['verdict']}")
    return "\n".join(lines)
