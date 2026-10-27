"""Ingest: turn a paper or post into `kind: source` claims, each with the source's own tests.

Two Claude calls. Claims come from the abstract (for a post, its whole text), where authors
state what they found. Then, for each claim, the body is read for the experiments the source
offers as evidence: what was done, where, on which models and data, with which metric and
numbers. Everything checkable is checked here without a model: each quote must appear in the
text it was taken from, and each stated number in the body. Claims land as `proposed`,
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

MODEL = "claude-sonnet-5-5"  # extraction: structured, and every quote/number is checked after
EFFORT = "medium"
EVIDENCE = ["observation", "probe", "ablation_or_patching", "steering", "causal_abstraction",
            "derivation", "benchmark", "qualitative_examples", "none"]
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
        "references": {"type": "object", "additionalProperties": False, "required": ["code", "data", "models"],
                       "properties": {k: {"type": "array", "items": {"type": "string"}} for k in ("code", "data", "models")}},
    },
}

CLAIMS_PROMPT = """You are extracting claims from an interpretability {kind} for a claim database that research agents reason over. Authors state their findings in the abstract; extract each distinct finding there as one claim (typically 3-8 for a paper, 1-3 for a post). Keep the authors' strength of language in `hedge` ("argues" for conclusions drawn from other findings). `quote` must be copied character for character from the text below; it will be checked.

{ref} — {heading}:
<text>
{text}
</text>"""

TESTS_PROMPT = """These claims were extracted from the abstract of {ref}. For each claim, find in the body the experiments the authors offer as evidence for it, and describe each concretely enough that someone could rerun it: what was done, the models, the data, the metric, where it is (section / figure / table / appendix), and every number the authors report for it, copied exactly as written. `quote` must be copied character for character from the body; it will be checked. If the body offers no experiment for a claim, return an empty list for it. Also list the code repositories, datasets and model identifiers the source names.

CLAIMS:
{claims}

BODY:
<body>
{body}
</body>"""

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


def fetch_abstract(arxiv_id: str) -> tuple[str, str, list[str]]:
    """(abstract, first-version date, authors) from the arXiv API, not from the model."""
    arxiv_id = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|html|pdf)/)", "", arxiv_id, flags=re.I).strip("/")
    with urllib.request.urlopen(f"https://export.arxiv.org/api/query?id_list={arxiv_id}", timeout=60) as r:
        xml = r.read().decode("utf-8", "replace")
    m = re.search(r"<entry>.*?<summary>(.*?)</summary>", xml, re.S)
    d = re.search(r"<entry>.*?<published>(\d{4}-\d{2}-\d{2})", xml, re.S)
    entry = xml.split("<entry>", 1)[-1]
    authors = [html.unescape(a.strip()) for a in re.findall(r"<author>\s*<name>(.*?)</name>", entry, re.S)]
    return (re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else "", d.group(1) if d else "", authors)


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


def extract(abstract: str, body: str, kind: str, ref: str, client=None) -> dict:
    """Claims from the abstract, then each claim's tests from the body."""
    heading = "abstract" if kind == "paper" else "full text"
    out = _call(CLAIMS_PROMPT.format(kind=kind, ref=ref, heading=heading, text=abstract), CLAIMS_SCHEMA, client)
    claims = [{"slug": c["slug"], "statement": c["statement"]} for c in out["claims"]]
    tests = _call(TESTS_PROMPT.format(ref=ref, claims=json.dumps(claims, indent=1), body=body), TESTS_SCHEMA, client)
    by_slug = {t["slug"]: t for t in tests["claims"]}
    for c in out["claims"]:
        t = by_slug.get(c["slug"], {"scope": "", "tests": []})
        c["scope"], c["tests"] = t["scope"], t["tests"]
    out["references"] = tests["references"]
    return out


def check(abstract: str, body: str, claim: dict) -> dict:
    """Quote in the abstract; per test, quote and each stated value in the body."""
    a, b = _norm(abstract), _norm(body)
    return {"quote_found": bool(claim["quote"]) and _norm(claim["quote"]) in a,
            "tests": [{"quote_found": bool(t["quote"]) and _norm(t["quote"]) in b,
                       "values_found": {v["value"]: _norm(v["value"]) in b for v in t["stated_values"]}}
                      for t in claim.get("tests", [])]}


EVIDENCE_MAP = {"observation": "IOB", "probe": "ICP", "ablation_or_patching": "IMP", "steering": "ISS",
                "causal_abstraction": "ICA"}
STRENGTH = ["IOB", "ICP", "IMP", "ISS", "ICA"]


def to_claims(out: dict, abstract: str, body: str, kind: str, ref: str, prefix: str, question: str | None) -> list[dict]:
    claims = []
    for c in out["claims"]:
        codes = [EVIDENCE_MAP[t["evidence"]] for t in c["tests"] if t["evidence"] in EVIDENCE_MAP]
        src = {"kind": kind, "ref": ref, "title": out["title"], "locator": "Abstract" if kind == "paper" else "post",
               "scope": c["scope"]}
        if out["date"]:
            src["date"] = out["date"]
        if (out.get("references") or {}).get("code"):
            src["code"] = out["references"]["code"]
        if codes:
            src["stated_evidence"] = max(codes, key=STRENGTH.index)
        d = {"id": f"src.{prefix}.{c['slug']}", "kind": "source", "evidence": "TAS" if kind == "paper" else "NAS",
             "status": "proposed", "hedge": c["hedge"], "statement": c["statement"], "quote": c["quote"]}
        if question:
            d["question"] = question
        if c["tests"]:
            d["stated"] = {"tests": c["tests"]}
        if kind != "paper":
            del src["locator"]
        d["provenance"] = {"source": src, "assigned_by": f"agent:ingest ({MODEL})", "reviewed": False,
                           "checks": check(abstract, body, c)}
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
