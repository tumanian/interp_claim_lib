"""Source in, claims and tests out: ingest -> match -> write -> instantiate templates.

Used by `claimci ingest` and the web app. Model runs are separate (`verify`, locally or
on a remote GPU box): this step only calls the Claude API and writes files.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import ingest, match, templates
from .loader import load_repo
from .validate import validate


def instantiate(source: dict, m: dict) -> tuple[list[dict], str | None]:
    """The tests a template match produces for one source claim, or why there are none."""
    problem = match.check_params(m) if m else None
    if not m or not m.get("template") or problem:
        return [], problem or "no test template fits this claim yet"
    if m["template"] == "neuron_steering":
        return templates.neuron_steering(source, m["hf_model"], m["interventions"]), None
    if m["template"] == "contrastive_direction":
        return [t for p in m["packs"] for t in templates.contrastive_direction(source, m["hf_model"], int(m["layer"]), p)], None
    return [], f"template {m['template']} has no instantiation"


def match_source(root: Path, ref: str, log=print) -> dict:
    """Run matching for source claims already in the base (no re-extraction) and write the tests
    their templates produce. Existing claims are never overwritten."""
    root = Path(root)
    repo = load_repo(root)
    mine = {k: c for k, c in repo.claims.items() if c.is_source and (c.provenance.get("source") or {}).get("ref") == ref}
    if not mine:
        raise KeyError(f"no source claims for {ref}")
    others = {k: c for k, c in repo.claims.items() if c.is_source and k not in mine}
    claims = [{"id": k, "statement": c.statement, "quote": c.quote, "stated": c.stated, "question": c.question}
              for k, c in sorted(mine.items())]
    log(f"matching {len(claims)} source claims of {ref} against templates and {len(others)} other source claims")
    matches = match.match(claims, others, ref)
    out, made = [], []
    for d in claims:
        m = matches.get(d["id"], {})
        tests, problem = instantiate(d, m) if not m.get("same_as") else ([], None)
        made += tests
        out.append({"id": d["id"], "same_as": m.get("same_as") or None, "template": m.get("template") or None,
                    "params": {k: m.get(k) for k in ("hf_model", "packs", "layer", "interventions") if m.get(k) not in (None, "", -1, [])},
                    "tests": [t["id"] for t in tests], "gap": problem, "why": m.get("why")})
    written = templates.write_claims(made, root / "claims", root / "questions")
    errors = validate(load_repo(root))
    log(f"wrote {len(written)} tests" + (f"; validation errors: {errors}" if errors else ""))
    return {"ref": ref, "claims": out, "tests_written": written, "errors": errors}


def ingest_source(root: Path, source: str, text: str | None = None, question: str | None = None,
                  do_match: bool = True, log=print) -> dict:
    root = Path(root)
    if text is not None:  # a post: its whole text is both where claims and evidence are stated
        kind, ref, abstract, body = "post", source, text, text
    else:
        log(f"fetching {source}")
        body, ref = ingest.fetch_arxiv(source)
        (abstract, date, authors), kind = ingest.fetch_abstract(source), "paper"
    log(f"extracting claims from the {'abstract' if kind == 'paper' else 'post'} with {ingest.MODEL}, "
        f"then each claim's tests from the body ({len(body):,} characters)")
    out = ingest.extract(abstract, body, kind, ref)
    if kind == "paper" and date:
        out["date"] = date  # from arXiv, not from the model
    if kind == "paper" and authors:
        out["authors"] = authors
    prefix = ingest.prefix_for(out, ref)
    if question is None:
        question = f"q.{prefix}"
        qp = root / "questions" / f"{question}.yaml"
        if not qp.exists():
            qp.write_text(yaml.safe_dump({"id": question, "text": f"What does {out['title']} claim, and does it hold?",
                                          "status": "open", "claims": []}, sort_keys=False))
    claims = ingest.to_claims(out, abstract, body, kind, ref, prefix, question)
    log(f"{out['title']}: {len(claims)} claims")
    repo = load_repo(root)
    existing = {k: c for k, c in repo.claims.items() if c.is_source}
    matches = {}
    if do_match and claims:
        log("matching against existing source claims and test templates")
        matches = match.match(claims, existing, ref)
    summary = {"title": out["title"], "ref": ref, "kind": kind, "references": out["references"],
               "question": question, "claims": []}
    to_write, instantiated = [], []
    for d in claims:
        m = matches.get(d["id"], {})
        entry = {"id": d["id"], "statement": d["statement"], "hedge": d["hedge"],
                 "paper_tests": len((d.get("stated") or {}).get("tests", [])),
                 "quote_found": d["provenance"]["checks"]["quote_found"], "why": m.get("why")}
        if m.get("same_as"):
            entry.update(status="same_as", same_as=m["same_as"])
        else:
            to_write.append(d)
            tests, problem = instantiate(d, m)
            instantiated += tests
            entry.update(status="new", **({"template": m["template"], "tests": [t["id"] for t in tests]} if tests
                                          else {"gap": problem}))
        summary["claims"].append(entry)
    written, skipped = ingest.write(to_write, root / "claims")
    tests_written = templates.write_claims(instantiated, root / "claims", root / "questions")
    errors = validate(load_repo(root))
    summary.update(written=written, skipped=skipped, tests_written=tests_written, errors=errors)
    log(f"wrote {len(written)} source claims and {len(tests_written)} tests"
        + (f"; validation errors: {errors}" if errors else ""))
    return summary
