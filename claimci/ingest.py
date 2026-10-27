"""Ingest: turn a paper or post into `kind: source` claims with checked quotes.

Claude reads the source and proposes claims; everything it says that can be checked
mechanically is checked here, without a model: each quote must appear in the source
text, and each stated number must appear in it too. Claims land as `proposed`, marked
machine-made (`assigned_by: agent:ingest`) and unreviewed, with the check results.
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import yaml

MODEL = "claude-opus-5-5"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "authors", "date", "claims", "references"],
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "date": {"type": "string", "description": "YYYY-MM-DD if known, else empty"},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["slug", "statement", "quote", "locator", "hedge", "evidence_offered", "scope", "stated_values"],
            "properties": {
                "slug": {"type": "string", "description": "snake_case, 2-5 words, unique within the source"},
                "statement": {"type": "string", "description": "the assertion, in one or two plain sentences"},
                "quote": {"type": "string", "description": "a verbatim span from the source (under 40 words) that states it"},
                "locator": {"type": "string", "description": "section / figure / table, or empty if none"},
                "hedge": {"type": "string", "enum": ["shows", "suggests", "speculates", "argues"]},
                "evidence_offered": {"type": "string", "enum": [
                    "observation", "probe", "ablation_or_patching", "steering", "causal_abstraction",
                    "derivation", "benchmark", "none", "argument_from_other_claims"]},
                "scope": {"type": "string", "description": "models, data and settings the source actually studied"},
                "stated_values": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "value", "context"],
                    "properties": {"name": {"type": "string"}, "value": {"type": "string"},
                                   "context": {"type": "string"}}}},
            }}},
        "references": {
            "type": "object", "additionalProperties": False,
            "required": ["code", "data", "models"],
            "properties": {k: {"type": "array", "items": {"type": "string"}} for k in ("code", "data", "models")}},
    },
}

PROMPT = """You are extracting claims from an interpretability source for a claim database \
that research agents reason over. Extract the source's own claims: one per assertion that the \
source backs with its own evidence (a section, figure, table or derivation), plus the authors' \
main conclusions marked hedge "argues" when they rest on other claims. Do not extract claims the \
source only cites from prior work. Keep the authors' strength of language in `hedge`. \
`quote` must be copied character for character from the text below; it will be checked. \
List every numeric result the source states for a claim in `stated_values`, with the value \
exactly as written. In `references`, list code repositories, datasets and model identifiers \
the source names. Aim for 5-15 claims for a paper and 1-3 for a short post.

SOURCE ({kind}, {ref}):
<source>
{text}
</source>"""


class _Text(HTMLParser):
    """arXiv HTML (LaTeXML) to plain text, headings marked with '## '."""
    SKIP = {"script", "style", "nav", "header", "footer", "button"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5"):
            self.out.append("\n\n## ")
        elif tag in ("p", "div", "li", "tr", "br", "figcaption"):
            self.out.append("\n")
        elif tag == "math":
            alt = dict(attrs).get("alttext")
            if alt:
                self.out.append(f" {alt} ")
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP or tag == "math":
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def fetch_arxiv(arxiv_id: str) -> tuple[str, str]:
    """(text, ref) for an arXiv id, from the HTML rendering."""
    arxiv_id = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|html|pdf)/)", "", arxiv_id, flags=re.I).strip("/")
    with urllib.request.urlopen(f"https://arxiv.org/html/{arxiv_id}", timeout=60) as r:
        p = _Text()
        p.feed(r.read().decode("utf-8", "replace"))
    text = re.sub(r"[ \t]+", " ", html.unescape("".join(p.out)))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip(), f"arXiv:{re.sub(r'v[0-9]+$', '', arxiv_id)}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def check(text: str, claim: dict) -> dict:
    """Mechanical checks against the source text: quote present, each stated value present."""
    t = _norm(text)
    values = {v["value"]: _norm(v["value"]) in t for v in claim["stated_values"]}
    return {"quote_found": bool(claim["quote"]) and _norm(claim["quote"]) in t,
            "values_found": values}


def extract(text: str, kind: str, ref: str, client=None) -> dict:
    import anthropic
    client = client or anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL, max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(kind=kind, ref=ref, text=text)}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"extraction declined: {getattr(msg.stop_details, 'category', None)}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("extraction hit max_tokens; the JSON is incomplete")
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


EVIDENCE_MAP = {"observation": "IOB", "probe": "ICP", "ablation_or_patching": "IMP", "steering": "ISS",
                "causal_abstraction": "ICA"}


def to_claims(out: dict, text: str, kind: str, ref: str, prefix: str, question: str | None) -> list[dict]:
    claims = []
    for c in out["claims"]:
        checks = check(text, c)
        src = {"kind": kind, "ref": ref, "title": out["title"], "scope": c["scope"]}
        if c["locator"]:
            src["locator"] = c["locator"]
        if out["date"]:
            src["date"] = out["date"]
        if c["evidence_offered"] in EVIDENCE_MAP:
            src["stated_evidence"] = EVIDENCE_MAP[c["evidence_offered"]]
        d = {"id": f"src.{prefix}.{c['slug']}", "kind": "source", "evidence": "TAS" if c["locator"] else "NAS",
             "status": "proposed", "hedge": c["hedge"], "statement": c["statement"], "quote": c["quote"]}
        if question:
            d["question"] = question
        if c["stated_values"]:
            d["stated"] = {"values": c["stated_values"]}
        d["provenance"] = {"source": src, "assigned_by": f"agent:ingest ({MODEL})", "reviewed": False,
                           "checks": checks}
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
