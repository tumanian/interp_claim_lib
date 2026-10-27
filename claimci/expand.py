"""Expand a source's verifications to other models: same template, same data, new model.

Every template-generated claim records its template and parameters (provenance.template), so
nothing is re-extracted or re-matched: the template is instantiated again with another model.
Layer-specific parameters are moved to the same relative depth, and that choice is recorded on
the new claims as supplemented (it is ours, not the paper's).

    claimci expand arXiv:2507.21509 --models Qwen/Qwen3.5-4B,meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import json
from pathlib import Path

from . import templates
from .loader import load_repo
from .validate import validate


# Published layer counts for gated models whose config cannot be read without access.
KNOWN_LAYERS = {"meta-llama/Llama-2-7b-chat-hf": 32, "meta-llama/Llama-2-13b-chat-hf": 40,
                "meta-llama/Llama-2-7b-hf": 32, "meta-llama/Meta-Llama-3-8B-Instruct": 32,
                "meta-llama/Llama-3.1-8B-Instruct": 32, "google/gemma-2b-it": 18, "google/gemma-7b-it": 28}


def n_layers(hf_model: str) -> int:
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.loads(Path(hf_hub_download(hf_model, "config.json")).read_text())
    except Exception:  # gated without access: fall back to the published architecture
        if hf_model in KNOWN_LAYERS:
            return KNOWN_LAYERS[hf_model]
        raise
    return int((cfg.get("text_config") or cfg)["num_hidden_layers"])


def scale_block(block: int, src_layers: int, dst_layers: int) -> int:
    """The block at the same relative depth: blocks are 0..n-1, depth measured to the block's output."""
    return max(0, min(dst_layers - 1, round((block + 1) / src_layers * dst_layers) - 1))


def plan(root: Path, ref: str) -> list[dict]:
    """The template instances behind a source's verifications: one per (template, params)."""
    repo = load_repo(root)
    mine = {k for k, c in repo.claims.items() if c.is_source and (c.provenance.get("source") or {}).get("ref") == ref}
    seen, out = set(), []
    for k, c in sorted(repo.claims.items()):
        t = c.provenance.get("template")
        if c.is_source or not t or (c.about not in mine and (c.stated or {}).get("from") not in mine):
            continue
        key = json.dumps(t, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append({"template": t, "source": c.about or c.stated.get("from")})
    return out


def expand(root: Path, ref: str, models: list[str], log=print) -> dict:
    root = Path(root)
    repo = load_repo(root)
    made, skipped = [], []
    for inst in plan(root, ref):
        t, src = inst["template"], repo.claims[inst["source"]]
        source = {"id": src.id, "question": src.question}
        if t["name"] != "contrastive_direction":
            skipped.append(f"{t['name']}: expand not supported yet")
            continue
        p = t["params"]
        src_n = n_layers(p["hf_model"])
        for m in models:
            if m == p["hf_model"]:
                continue
            dst_n = n_layers(m)
            block = scale_block(p["block"], src_n, dst_n)
            note = (f"expanded from {p['hf_model']} (block {p['block']} of {src_n}) to block {block} of {dst_n} "
                    f"by relative depth; the paper did not test this model")
            claims = templates.contrastive_direction(source, m, p["layer"], p["pack"], block=block, note=note,
                                                     expanded_from=p["hf_model"])
            log(f"{p['pack']} on {m}: block {block} of {dst_n} ({len(claims)} claims)")
            made += claims
    written = templates.write_claims(made, root / "claims", root / "questions")
    errors = validate(load_repo(root))
    return {"written": written, "skipped": skipped, "errors": errors}
