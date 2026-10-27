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


# --- report card: the first thing shown for a source ---------------------------------------

EVIDENCE_LABELS = {  # strength, method: shown as a two-part pill
    "IMP": ("Causal", "Ablation / patching"), "ISS": ("Causal", "Activation steering"),
    "ICA": ("Causal", "Interchange intervention"), "IOB": ("Correlational", "Activation reading"),
    "ICP": ("Correlational", "Linear probe"), "IEA": ("Automated", "Auto-interp label"),
    "TAS": ("Stated", "Author statement, cited"), "NAS": ("Stated", "Author statement"),
}
KIND_LABELS = {"source": "Finding", "test": "Verification", "measurement": "Readout"}
FINDING_STATUS = {  # card support -> one word for the report card
    "supported": "Verified", "contradicted by a test": "Not supported",
    "not supported: a control reproduced the effect": "Not supported", "partly tested": "Partly verified",
    "tests attached, not run": "Not yet run", "untested": "Untested"}


def _data_sources(c) -> list[dict]:
    """Human names for a claim's data: each vendored file's own `source` line."""
    import json
    from .datasets import DATA_DIR
    files = ((c.data.get("dataset") or {}).get("params") or {}).get("files") or {}
    out = []
    for name, rel in sorted(files.items()):
        src = ""
        if rel.endswith(".json"):
            try:
                v = json.loads((DATA_DIR / rel).read_text()).get("source")
                src = (f"{v['repo'].removeprefix('https://')}@{v['commit'][:7]}" if isinstance(v, dict) and "repo" in v
                       else str(v or ""))
            except (OSError, ValueError, AttributeError):
                pass
        elif rel.endswith(".pt"):
            src = "published artifact from the source's repository"
        label = {"prompts": "the source's own prompt samples", "reference_direction": "the authors' published direction",
                 "text": "natural text (wikitext-2)", "persona": "persona questions (Anthropic model-written evals)",
                 "em": "Emergent Misalignment questions and judge prompts"}.get(name, name)
        out.append({"name": label, "file": rel, "source": src})
    return out


def report(repo: Repo, ledger: Ledger, ref: str) -> dict:
    """Report card for one source: findings with one-word status, and what they were tested on."""
    findings = explain(repo, ledger, ref)
    if not findings:
        return {}
    s = findings[0]["source"]
    tested = {k for f in findings for a in f["attached"] for k in [a["id"]] + [x["id"] for x in a["controls"]]}
    q = {f["question"] for f in findings}
    tested |= {k for k, c in repo.claims.items() if not c.is_source and c.question in q}
    claims = [repo.claims[k] for k in sorted(tested)]
    models = sorted({c.model for c in claims})
    data, seen = [], set()
    for c in claims:
        for d in _data_sources(c):
            if d["file"] not in seen:
                seen.add(d["file"])
                data.append(d)
    code = {}
    for c in claims:
        ep = c.code.get("entrypoint", "")
        origin = (f"generated from template {c.provenance['assigned_by'].split()[-1]}"
                  if str(c.provenance.get("assigned_by", "")).startswith("template") else "written in the claimci library")
        code.setdefault(ep.split(":")[0], {"module": ep.split(":")[0], "origin": origin, "functions": set()})["functions"].add(ep.split(":")[-1])
    pulled = sorted({d["source"] for d in data if "github.com" in d["source"] and "refusal_direction" in d["file"]})
    for f in findings:
        f["status"] = FINDING_STATUS.get(f["support"], f["support"])
        f["evidence_label"] = EVIDENCE_LABELS.get(f["evidence"])
        for a in f["attached"]:
            a["kind_label"], a["evidence_label"] = KIND_LABELS[a["kind"]], EVIDENCE_LABELS.get(a["evidence"])
            a["judged"] = repo.claims[a["id"]].params.get("judge_model") if "judge" in repo.claims[a["id"]].code.get("entrypoint", "") else None
            a["supplemented"] = sorted((repo.claims[a["id"]].provenance.get("supplemented") or {}).items())
    counts = {}
    for f in findings:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    order = {"Verified": 0, "Not supported": 1, "Partly verified": 2, "Not yet run": 3, "Untested": 4}
    findings.sort(key=lambda f: (order.get(f["status"], 9), f["id"]))
    return {"ref": ref, "title": s.get("title") or ref, "kind": s.get("kind"), "date": s.get("date"),
            "counts": counts, "n_findings": len(findings), "models": models, "data": data,
            "code": [{**v, "functions": sorted(v["functions"])} for v in code.values()], "pulled_from_source": pulled,
            "findings": findings}
