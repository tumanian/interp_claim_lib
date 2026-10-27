"""Expand a source's verifications to other models: dumb machinery over claims.

    claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3.5-4B [--verify]

For each source finding, take the claims about it on their original model, with their dependency
closure and controls, and clone them onto each target model. Nothing here knows any template or
experiment: each parameter is moved by the kind its experiment declares (@experiment params):

  layer      scaled to the same relative depth            recorded as supplemented
  component  neuron / head indices, model-specific        the closure is refused, with the reason
  tokenizer  token ids, tokenizer-specific                the closure is refused, with the reason
  input      bound from an upstream claim                 rewired to the cloned upstream claim
  data, scalar                                            unchanged

Clones are marked `expanded_from`, so cards report them as "beyond the paper" and they never count
against the source. With --verify they run through the executor (claimci/run/executors.py).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml

from ..schema.experiments import lock
from ..schema.loader import load_repo
from ..checks import validate

# Published layer counts for gated models whose config cannot be read without access.
KNOWN_LAYERS = {"meta-llama/Llama-2-7b-chat-hf": 32, "meta-llama/Llama-2-13b-chat-hf": 40,
                "meta-llama/Llama-2-7b-hf": 32, "meta-llama/Meta-Llama-3-8B-Instruct": 32,
                "meta-llama/Llama-3.1-8B-Instruct": 32, "google/gemma-2b-it": 18, "google/gemma-7b-it": 28}
MOVABLE = {"layer", "input", "data", "scalar"}


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


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.split("/")[-1].lower())


def default_opts(hf_model: str, like: dict) -> dict:
    """Model options for a clone: keep how the original ran (chat, dtype, thinking) but not what was
    specific to its model (a paper's hand-written template, its batch size)."""
    opts = {k: v for k, v in like.items() if k in ("chat", "dtype", "thinking")}
    opts.update(template="hf", batch=1 if "qwen3.5" in hf_model.lower() else 16)  # linear attention: no padding
    return opts


def closures(repo, ref: str) -> list[tuple[str, list[str]]]:
    """(original model, claim ids) per source finding: the claims about it that were not themselves
    expansions, with everything they depend on and their controls."""
    findings = {k for k, c in repo.claims.items() if c.is_source and (c.provenance.get("source") or {}).get("ref") == ref}
    out = []
    for f in sorted(findings):
        roots = {k for k, c in repo.claims.items() if c.about == f and not c.provenance.get("expanded_from")}
        if not roots:
            continue
        ids = set(roots)
        for r in roots:
            ids |= repo.ancestors(r)
        ids |= {k for k, c in repo.claims.items() if c.control_for in ids and not c.provenance.get("expanded_from")}
        by_model: dict[str, list[str]] = {}
        for k in sorted(ids):
            by_model.setdefault(repo.claims[k].model, []).append(k)
        out += sorted(by_model.items())
    return out


def clone(repo, ids: list[str], src_model: str, dst_model: str, layers: tuple[int, int]) -> tuple[list[dict], str | None]:
    """Clone claim files onto dst_model, or ([], reason) when a parameter cannot move."""
    reg = lock()
    s_old, s_new = _slug(src_model), _slug(dst_model)
    rename = {k: (k.replace(s_old, s_new) if s_old in k else f"{k}.{s_new}") for k in ids}
    out = []
    for k in ids:
        c = repo.claims[k]
        kinds = reg[c.experiment_id]["params"]
        d = copy.deepcopy(yaml.safe_load(c.path.read_text()))
        notes = {}
        for p, v in (d.get("code", {}).get("params") or {}).items():
            kind = kinds.get(p)
            if kind not in MOVABLE:
                why = "undeclared" if kind is None else f"a {kind} parameter"
                return [], f"{k}: `{p}` is {why}, specific to {src_model}; this experiment cannot move to another model"
            if kind == "layer":
                d["code"]["params"][p] = scale_block(int(v), *layers)
                notes[f"code.params.{p}"] = {"by": "expand", "note": f"{p} {v} of {layers[0]} layers in {src_model} -> "
                                             f"{d['code']['params'][p]} of {layers[1]} in {dst_model}, by relative depth"}
        d["id"] = rename[k]
        d["subject"]["model"] = {"hf": dst_model, "revision": "main", **default_opts(dst_model, c.model_opts)}
        d["depends_on"] = [rename.get(x, x) for x in d.get("depends_on") or []]
        d["inputs"] = {n: (rename[r.rsplit(".", 1)[0]] + "." + r.rsplit(".", 1)[1]) if r.rsplit(".", 1)[0] in rename else r
                       for n, r in (d.get("inputs") or {}).items()}
        if (d.get("data") or {}).get("inherit") in rename:
            d["data"]["inherit"] = rename[d["data"]["inherit"]]
        if d.get("control_for") in rename:
            d["control_for"] = rename[d["control_for"]]
        prov = d.setdefault("provenance", {})
        prov["expanded_from"] = src_model
        prov.setdefault("supplemented", {}).update(notes)
        prov["supplemented"]["subject.model"] = {"by": "expand", "note": f"moved from {src_model}; the source did not test {dst_model}"}
        out.append(d)
    return out, None


def write(claims: list[dict], root: Path) -> list[str]:
    written = []
    for d in claims:
        p = root / "claims" / f"{d['id']}.yaml"
        if not p.exists():
            p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100))
            written.append(d["id"])
    for qid in {d.get("question") for d in claims if d.get("question")}:
        qp = root / "questions" / f"{qid}.yaml"
        if qp.exists():
            q = yaml.safe_load(qp.read_text())
            q["claims"] = list(dict.fromkeys(list(q.get("claims") or []) + [d["id"] for d in claims if d.get("question") == qid]))
            qp.write_text(yaml.safe_dump(q, sort_keys=False, allow_unicode=True, width=100))
    return written


def expand(root: Path, ref: str, models: list[str], verify: bool = False, log=print) -> dict:
    root = Path(root)
    repo = load_repo(root)
    made, refused = [], []
    for src_model, ids in closures(repo, ref):
        for m in models:
            if m == src_model:
                continue
            claims, why = clone(repo, ids, src_model, m, (n_layers(src_model), n_layers(m)))
            if why:
                refused.append(why)
                log(f"refused: {why}")
                continue
            log(f"{len(claims)} claims from {src_model} -> {m}")
            made += claims
    written = write(made, root)
    errors = validate(load_repo(root))
    out = {"written": written, "refused": refused, "errors": errors}
    if verify and written and not errors:
        from ..run.executors import executor_for
        out["verify_exit"] = executor_for(root).verify(written, log)
    return out
