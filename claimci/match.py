"""Match newly ingested source claims against the base: duplicates and test templates.

One Claude call per ingest. For each new claim it answers two questions:
  same_as   - is this the same assertion as a source claim already in the base? Then the
              new one is not written; the existing claim (and its tests) is reported instead.
  template  - if not, does a test template fit, with every parameter stated by the source?
              Then the template's tests are instantiated. Otherwise the claim is a gap.
Parameters are checked here before use (the model id must exist on Hugging Face).
"""
from __future__ import annotations

import json

from .ingest import EFFORT, MODEL
from .templates import CATALOG

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["matches"],
    "properties": {"matches": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["slug", "same_as", "template", "hf_model", "interventions", "why"],
        "properties": {
            "slug": {"type": "string"},
            "same_as": {"type": "string", "description": "existing source claim id, or empty"},
            "template": {"type": "string", "enum": ["", *CATALOG]},
            "hf_model": {"type": "string", "description": "Hugging Face repo id, e.g. Qwen/Qwen3.5-4B, or empty"},
            "interventions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["layer", "neuron", "dose"],
                "properties": {"layer": {"type": "integer"}, "neuron": {"type": "integer"}, "dose": {"type": "number"}}}},
            "why": {"type": "string", "description": "one sentence"}}}}},
}

PROMPT = """New source claims were just extracted from {ref}. For each one decide:

1. same_as: the id of an EXISTING source claim that makes the same assertion about the same \
thing (same source or another source making the identical claim). Leave empty if none; a related \
but different claim is not the same.
2. template: a test template from the catalog that fits, but only if the claim states every \
parameter the template needs. Never guess a parameter. Leave empty otherwise.
3. hf_model and interventions: the template's parameters exactly as the source states them \
(empty if no template). Map an informal model name to its official Hugging Face repo id.

TEMPLATE CATALOG:
{catalog}

EXISTING SOURCE CLAIMS:
{existing}

NEW CLAIMS:
{new}"""


def match(new_claims: list[dict], existing: dict, ref: str, client=None) -> dict[str, dict]:
    import anthropic
    client = client or anthropic.Anthropic()
    new = [{"slug": c["id"].split(".")[-1], "id": c["id"], "statement": c["statement"],
            "quote": c.get("quote"), "stated": c.get("stated")} for c in new_claims]
    ex = [{"id": k, "statement": " ".join(c.statement.split()),
           "source": (c.provenance.get("source") or {}).get("ref")} for k, c in sorted(existing.items())]
    prompt = PROMPT.format(ref=ref, catalog=json.dumps(CATALOG, indent=1), existing=json.dumps(ex, indent=1),
                           new=json.dumps(new, indent=1))
    with client.beta.messages.stream(
        model=MODEL, max_tokens=32000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"matching stopped: {msg.stop_reason}")
    out = json.loads(next(b.text for b in msg.content if b.type == "text"))
    by_slug = {c["id"].split(".")[-1]: c["id"] for c in new_claims}
    result = {}
    for m in out["matches"]:
        cid = by_slug.get(m["slug"])
        if cid is None:
            continue
        if m["same_as"] and m["same_as"] not in existing:
            m["same_as"] = ""  # an id the model invented is ignored
        result[cid] = m
    return result


def check_params(m: dict) -> str | None:
    """Why a template match cannot be instantiated, or None if it can."""
    if m["template"] != "neuron_steering":
        return "template not instantiable automatically" if m["template"] else None
    if not m["hf_model"] or not m["interventions"]:
        return "missing model or interventions"
    try:
        from huggingface_hub import model_info
        model_info(m["hf_model"])
    except Exception as e:  # unknown repo, gated without access, network
        return f"model {m['hf_model']} not available on Hugging Face ({type(e).__name__})"
    return None
