"""Extraction, shared by every source: findings from where the source states them, then each
finding's tests from where its evidence is, then mechanical checks, then claims.

Two Claude calls per source (Sonnet at medium effort). Everything checkable is checked without a
model: each finding's quote must appear in the text it was taken from (the Document's claims_text),
each test's quote and stated numbers in the body. Claims land as `proposed`, machine-made
(`assigned_by: agent:ingest`) and unreviewed, with the check results. The schemas below shape
Claude's output only; what is stored is mapped onto claimci/schema/source_tests.py explicitly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ..schema.source_tests import EVIDENCE as SOURCE_EVIDENCE, SourceTest
from .document import Document, normalize

MODEL = "claude-sonnet-5-5"  # extraction: structured, and every quote/number is checked after
EFFORT = "medium"
EVIDENCE = list(SOURCE_EVIDENCE)
_VALUE = {"type": "object", "additionalProperties": False, "required": ["name", "value", "context"],
          "properties": {"name": {"type": "string"}, "value": {"type": "string"}, "context": {"type": "string"}}}

CLAIMS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "authors", "date", "claims"],
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "date": {"type": "string", "description": "YYYY-MM-DD if known, else empty"},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["slug", "statement", "quote", "hedge"],
            "properties": {
                "slug": {"type": "string", "description": "snake_case, 2-5 words, unique within the source"},
                "statement": {"type": "string", "description": "the finding, in one or two plain sentences"},
                "quote": {"type": "string", "description": "verbatim span from the abstract (under 40 words)"},
                "hedge": {"type": "string", "enum": ["shows", "suggests", "speculates", "argues"]}}}}},
}

TESTS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims", "references"],
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["slug", "scope", "tests"],
            "properties": {
                "slug": {"type": "string"},
                "scope": {"type": "string", "description": "models, data and settings the source actually studied for it"},
                "tests": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["description", "locator", "evidence", "models", "data", "metric", "quote", "stated_values"],
                    "properties": {
                        "description": {"type": "string", "description": "the experiment, concretely enough to rerun"},
                        "locator": {"type": "string", "description": "section / figure / table / appendix"},
                        "evidence": {"type": "string", "enum": EVIDENCE},
                        "models": {"type": "array", "items": {"type": "string"}},
                        "data": {"type": "string"}, "metric": {"type": "string"},
                        "quote": {"type": "string", "description": "verbatim span from the body (under 40 words) reporting it"},
                        "stated_values": {"type": "array", "items": _VALUE}}}}}}},
        "references": {"type": "object", "additionalProperties": False, "required": ["own_code", "code", "data", "models"],
                       "properties": {"own_code": {"type": "array", "items": {"type": "string"},
                                                   "description": "repositories the authors release for THIS work (their code or data), not ones they cite"},
                                      **{k: {"type": "array", "items": {"type": "string"}} for k in ("code", "data", "models")}}},
    },
}

CLAIMS_PROMPT = """You are extracting claims from an interpretability {kind} for a claim database that research agents reason over. Authors state their findings in the {heading}; extract each distinct finding there as one claim (typically 3-8 for a paper, 1-3 for a post). Keep the authors' strength of language in `hedge` ("argues" for conclusions drawn from other findings). `quote` must be copied character for character from the text below; it will be checked.

{ref} — {heading}:
<text>
{text}
</text>"""

TESTS_PROMPT = """These claims were extracted from the {heading} of {ref}. For each claim, find in the body the experiments the authors offer as evidence for it, and describe each concretely enough that someone could rerun it: what was done, the models, the data, the metric, where it is (section / figure / table / appendix), and every number the authors report for it, copied exactly as written. `quote` must be copied character for character from the body; it will be checked. If the body offers no experiment for a claim, return an empty list for it. Also list the code repositories, datasets and model identifiers the source names, and separately (own_code) the \
repositories the authors release for this work.

CLAIMS:
{claims}

BODY:
<body>
{body}
</body>"""


def _call(prompt: str, schema: dict, client=None) -> dict:
    import anthropic
    client = client or anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL, max_tokens=64000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"extraction declined: {getattr(msg.stop_details, 'category', None)}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("extraction hit max_tokens; the JSON is incomplete")
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


HEADINGS = {"paper": "abstract", "blog": "summary", "post": "full text"}


def extract(doc: Document, client=None) -> dict:
    """Findings from doc.claims_text, then each finding's tests from doc.body."""
    heading = (doc.claims_locator or HEADINGS.get(doc.kind, "text")).lower()
    out = _call(CLAIMS_PROMPT.format(kind=doc.kind, ref=doc.ref, heading=heading, text=doc.claims_text), CLAIMS_SCHEMA, client)
    claims = [{"slug": c["slug"], "statement": c["statement"]} for c in out["claims"]]
    tests = _call(TESTS_PROMPT.format(ref=doc.ref, heading=heading, claims=json.dumps(claims, indent=1), body=doc.body), TESTS_SCHEMA, client)
    by_slug = {t["slug"]: t for t in tests["claims"]}
    for c in out["claims"]:
        t = by_slug.get(c["slug"], {"scope": "", "tests": []})
        c["scope"], c["tests"] = t["scope"], t["tests"]
    out["references"] = tests["references"]
    for k in ("title", "date"):         # metadata from the source itself wins over the model's reading
        if getattr(doc, k):
            out[k] = getattr(doc, k)
    if doc.authors:
        out["authors"] = doc.authors
    return out


def check(doc: Document, claim: dict) -> dict:
    """The finding's quote in claims_text; per test, its quote and each stated value in the body."""
    a, b = normalize(doc.claims_text), normalize(doc.body)
    return {"quote_found": bool(claim["quote"]) and normalize(claim["quote"]) in a,
            "tests": [{"quote_found": bool(t["quote"]) and normalize(t["quote"]) in b,
                       "values_found": {v["value"]: normalize(v["value"]) in b for v in t["stated_values"]}}
                      for t in claim.get("tests", [])]}


EVIDENCE_MAP = {"observation": "IOB", "probe": "ICP", "ablation_or_patching": "IMP", "steering": "ISS",
                "causal_abstraction": "ICA"}
STRENGTH = ["IOB", "ICP", "IMP", "ISS", "ICA"]


def to_claims(out: dict, doc: Document, prefix: str, question: str | None) -> list[dict]:
    claims = []
    for c in out["claims"]:
        tests = [SourceTest.from_dict(t).to_dict() for t in c["tests"]]   # the stored shape, not the prompt's
        codes = [EVIDENCE_MAP[t["evidence"]] for t in tests if t["evidence"] in EVIDENCE_MAP]
        src = {"kind": doc.kind, "ref": doc.ref, "title": out.get("title") or doc.title, "scope": c["scope"]}
        if doc.claims_locator:
            src["locator"] = doc.claims_locator
        if out.get("date"):
            src["date"] = out["date"]
        own = doc.own_code or (out.get("references") or {}).get("own_code") or []
        if own:  # the authors' own release, not everything they cite
            src["code"] = own
        if codes:
            src["stated_evidence"] = max(codes, key=STRENGTH.index)
        d = {"id": f"src.{prefix}.{c['slug']}", "kind": "source", "evidence": "TAS" if doc.claims_locator else "NAS",
             "status": "proposed", "hedge": c["hedge"], "statement": c["statement"], "quote": c["quote"]}
        if question:
            d["question"] = question
        if tests:
            d["stated"] = {"tests": tests}
        d["provenance"] = {"source": src, "assigned_by": f"agent:ingest ({MODEL})", "reviewed": False,
                           "checks": check(doc, c)}
        claims.append(d)
    return claims


def write(claims: list[dict], out_dir: Path) -> tuple[list[str], list[str]]:
    """Write new claim files; never overwrite an existing id (reviewed or hand-written claims win)."""
    written, skipped = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in claims:
        p = out_dir / f"{d['id']}.yaml"
        if p.exists():
            skipped.append(d["id"])
            continue
        p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100))
        written.append(d["id"])
    return written, skipped


def prefix_for(out: dict, ref: str) -> str:
    """e.g. arditi2024 from the first author's surname and year, else a slug of the ref."""
    year = (out.get("date") or "")[:4]
    if out.get("authors") and year:
        return re.sub(r"[^a-z]", "", out["authors"][0].split()[-1].lower()) + year
    return re.sub(r"[^a-z0-9]+", "_", ref.lower()).strip("_")[-30:]
