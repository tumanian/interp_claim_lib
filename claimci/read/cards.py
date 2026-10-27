"""Claim cards and source explanations: what an agent (or the web app) reads.

A card is a short, honest summary of one claim: what it measures, on what, against
which threshold and who set it, every result with its margin, and its controls. For a
source claim the card lists the tests and measurements attached to it, so "asserted by
the source" and "measured on this model" never look alike.
"""
from __future__ import annotations

from ..store.ledger import Ledger
from .report import computed_status, current_row, gated_by, revision_order
from ..schema.claim import OPS
from ..schema.repo import Repo


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
                    "value": r.value, "margin": c.margin(r.value, r.interval), "interval": r.interval, "gated_by": gated_by(repo, ledger, cid, rev),
                    "bindings": {k: v for k, v in r.bindings.items() if not str(v).startswith("artifact:")},
                    "reason": r.reason, "seconds": round(r.cost.get("seconds", 0.0), 1), "at": r.run.get("at"),
                    # judged runs keep every answer: the card offers them as samples to drill into
                    "samples": r.bindings.get("answers") if str(r.bindings.get("answers", "")).startswith("artifact:") else None,
                    "samples_n": r.bindings.get("n") or sum(p.get("n", 0) for p in r.bindings.get("per_seed") or []
                                                             if isinstance(p, dict)) or None})
    return out


def caveats(repo: Repo, ledger: Ledger, cid: str) -> list[str]:
    """Declared caveats that apply to the latest result at the subject revision. They qualify a
    verdict (\"verified, but ...\") and never change it."""
    c = repo.claims[cid]
    rows = [x for x in _results(repo, ledger, cid) if x["revision"] == c.revision]
    b = rows[-1]["bindings"] if rows else {}
    out = []
    for cv in c.caveats:
        v = b.get(cv["binding"])
        if isinstance(v, (int, float)) and OPS[cv["op"]](v, cv["value"]):
            out.append(cv["text"].format(value=v))
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
    d = {"id": cid, "kind": c.kind, "evidence": c.evidence, "evidence_label": " | ".join(EVIDENCE_LABELS.get(c.evidence, ())),
         "run_on": None if c.is_finding else ("expanded" if c.provenance.get("expanded_from") else "original"),
         "statement": " ".join(c.statement.split()),
         "question": c.question, "about": c.about, "status_in_file": c.status}
    if c.is_finding:
        d.update(hedge=c.hedge, level=c.level, quote=c.quote, source={k: src.get(k) for k in ("kind", "ref", "locator", "title", "date", "scope", "code") if src.get(k)},
                 machine_made=str(c.provenance.get("assigned_by", "")).startswith("agent:ingest"),
                 checks=c.provenance.get("checks"), stated=c.stated or None,
                 source_tests=[{**t, "check": (c.provenance.get("checks") or {}).get("tests", [{}] * 99)[i]}
                               for i, t in enumerate((c.stated or {}).get("tests", []))])
        attached = sorted(k for k, o in repo.claims.items() if o.about == cid and not o.control_for)
        d["attached"] = [{"id": k, "kind": repo.claims[k].kind, "evidence": repo.claims[k].evidence,
                          "statement": " ".join(repo.claims[k].statement.split()), "metric": repo.claims[k].metric,
                          "model": repo.claims[k].model, "threshold": _threshold(repo.claims[k]),
                          "verdict": verdict(repo, ledger, k), "results": _results(repo, ledger, k),
                          "caveats": caveats(repo, ledger, k), "post_hoc": bool(repo.claims[k].provenance.get("post_hoc")),
                          "controls": [{"id": j, "statement": " ".join(o.statement.split()),
                                        "threshold": _threshold(o), "verdict": verdict(repo, ledger, j),
                                        "post_hoc": bool(o.provenance.get("post_hoc")),
                                        "results": _results(repo, ledger, j)}
                                       for j, o in sorted(repo.claims.items()) if o.control_for == k]}
                         for k in attached]
        for a in d["attached"]:
            a["expansion"] = repo.claims[a["id"]].provenance.get("expanded_from")
        all_tests = [a for a in d["attached"] if a["kind"] == "test"]
        tests = [a for a in all_tests if not a["expansion"]]   # the source's own setting decides support
        d["caveats"] = [t for a in tests if a["verdict"] == "holds" for t in a["caveats"]]
        by_model: dict[str, list[str]] = {}
        for a in all_tests:
            if a["expansion"]:
                by_model.setdefault(a["model"], []).append(a["verdict"])
        d["expansions"] = [{"model": m, "held": sum(v == "holds" for v in vs),
                            "ran": sum(v in ("holds", "fails", "holds, control failed") for v in vs),
                            "could_not_run": sum(v in ("blocked", "error", "contaminated") for v in vs), "total": len(vs)}
                           for m, vs in sorted(by_model.items())]
        d["support"] = ("untested" if not all_tests else
                        "only tested beyond the source's setting" if not tests else
                        "not run on the source's model" if all(a["verdict"] in ("not run", "error", "blocked", "contaminated") for a in tests)
                        and any(a["verdict"] != "not run" for a in all_tests) else
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
    """Cards for every finding of one source: an arXiv id, a URL, a short name (chen2025, the part
    after src. in finding ids), or a single claim id."""
    if ref in repo.claims:
        return [card(repo, ledger, ref)]
    norm = ref.lower().replace("arxiv:", "").strip()
    ids = sorted(k for k, c in repo.claims.items() if c.is_finding
                 and (norm in str((c.provenance.get("source") or {}).get("ref", "")).lower() or k.startswith(f"src.{norm}.")))
    return [card(repo, ledger, k) for k in ids]


def render(d: dict) -> str:
    """Plain-text card, sized for an agent's context."""
    lines = [f"{d['id']}  [{d['kind']}, {d['evidence']}]" + (f"  about: {d['about']}" if d.get("about") else "")]
    lines.append(f"  {d['statement']}")
    if d["kind"] == "finding":
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
KIND_LABELS = {"finding": "Finding", "test": "Verification", "measurement": "Readout"}
FINDING_STATUS = {  # card support -> one word for the report card
    "supported": "Verified", "contradicted by a test": "Could not replicate",
    "not supported: a control reproduced the effect": "Effect not specific", "partly tested": "Partly verified",
    "tests attached, not run": "Not yet run", "untested": "Untested",
    "not run on the source's model": "Not run on the paper's model",
    "only tested beyond the source's setting": "Tested beyond the paper only"}


ORIGINS = ("from the authors", "pulled from elsewhere", "made here")


def _repo_key(url: str) -> str:
    """github.com/owner/name, lowercased, for comparing repository references."""
    import re
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url or "")
    return f"github.com/{m.group(1)}/{m.group(2).removesuffix('.git')}".lower() if m else ""


def _manifest() -> dict:
    import yaml
    from ..store.datasets import DATA_DIR
    p = DATA_DIR / "MANIFEST.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _data_items(c, author_repos: set[str]) -> list[dict]:
    """Each dataset a claim runs on, with where it came from and which of ORIGINS that is."""
    dv = c.dataset  # resolved, so claims that inherit their parent's data are covered
    if dv is None or dv.gen != "vendored":
        gen = dv.gen if dv else "?"
        return [{"key": f"gen:{gen}", "what": f"generated by {gen}", "from": "", "origin": "made here"}]
    out, man = [], _manifest()
    for name, rel in sorted((dv.params.get("files") or {}).items()):
        m = man.get(rel, {})
        src = m.get("from", "")
        origin = ("from the authors" if _repo_key(src) and _repo_key(src) in author_repos
                  else "pulled from elsewhere" if src else "made here")
        out.append({"key": rel, "what": m.get("what", rel), "from": src + (f"@{m['commit'][:7]}" if m.get("commit") else ""),
                    "license": m.get("license", ""), "origin": origin})
    return out


def _objects(repo, ledger, ids):
    from .objects import objects
    out = objects(repo, ledger, ids)
    for o in out:
        o["checks"] = [{"id": k, "kind_label": KIND_LABELS[repo.claims[k].kind],
                        "statement": " ".join(repo.claims[k].statement.split()), "verdict": verdict(repo, ledger, k)}
                       for k in o["claims"]]
    return out


def report(repo: Repo, ledger: Ledger, ref: str) -> dict:
    """Report card for one source: findings with one-word status, and what they were tested on."""
    findings = explain(repo, ledger, ref)
    if not findings:
        return {}
    s = findings[0]["source"]
    tested = {k for f in findings for a in f["attached"] for k in [a["id"]] + [x["id"] for x in a["controls"]]}
    q = {f["question"] for f in findings}
    tested |= {k for k, c in repo.claims.items() if not c.is_finding and c.question in q}
    claims = [repo.claims[k] for k in sorted(tested)]
    models = sorted({c.model for c in claims})
    # each model: original (the source used it) or expanded, with the model(s) it was carried over from
    model_runs = []
    for m in models:
        on = [c for c in claims if c.model == m]
        froms = sorted({c.provenance["expanded_from"] for c in on if c.provenance.get("expanded_from")})
        original = any(not c.provenance.get("expanded_from") for c in on)
        model_runs.append({"model": m, "run_on": "original" if original else "expanded", "from": [] if original else froms})
    model_runs.sort(key=lambda r: r["run_on"] != "original")   # the source's own model first
    author_repos = {_repo_key(u) for f in findings for u in (f["source"].get("code") or [])} - {""}
    data, seen = [], set()
    for c in claims:
        for d in _data_items(c, author_repos):
            if d["key"] not in seen:
                seen.add(d["key"])
                data.append(d)
    code = {}
    for c in claims:
        ep = c.entrypoint or ""
        mod = ep.split(":")[0]
        tmpl = str(c.provenance.get("assigned_by", "")).startswith("template")
        how = f"generated from template {c.provenance['assigned_by'].split()[-1]}" if tmpl else "written in the claimci library"
        e = code.setdefault(mod, {"what": mod, "origin": "made here", "how": set(), "functions": set()})
        e["how"].add(how)
        e["functions"].add(ep.split(":")[-1])
    code = [{**v, "how": "; ".join(sorted(v["how"])), "functions": sorted(v["functions"])} for v in code.values()]
    for url in sorted(author_repos):  # the authors' own code: listed, and said plainly whether we ran it
        code.append({"what": url, "origin": "from the authors", "how": "not run; the experiments above reimplement it",
                     "functions": []})
    for f in findings:
        f["status"] = FINDING_STATUS.get(f["support"], f["support"])
        f["evidence_label"] = EVIDENCE_LABELS.get(f["evidence"])
        for a in f["attached"]:
            a["kind_label"], a["evidence_label"] = KIND_LABELS[a["kind"]], EVIDENCE_LABELS.get(a["evidence"])
            a["judged"] = repo.claims[a["id"]].params.get("judge_model") if (repo.claims[a["id"]].experiment_id or "").startswith(("judge.", "persona.")) else None
            a["supplemented"] = sorted((repo.claims[a["id"]].provenance.get("supplemented") or {}).items())
    counts = {}
    for f in findings:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    order = {"Verified": 0, "Could not replicate": 1, "Effect not specific": 1, "Partly verified": 2, "Not yet run": 3, "Untested": 4}
    findings.sort(key=lambda f: (order.get(f["status"], 9), f["id"]))
    return {"ref": ref, "title": s.get("title") or ref, "kind": s.get("kind"), "date": s.get("date"),
            "counts": counts, "n_findings": len(findings), "models": models, "model_runs": model_runs, "data": data,
            "code": code, "origins": list(ORIGINS), "objects": _objects(repo, ledger, tested),
            "findings": findings}
