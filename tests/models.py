"""Model handles for tests. Never downloads.

real(): pythia-70m from the HF cache (preferred revision if cached), else None.
synthetic(): random-init GPT-NeoX with pythia-70m's architecture, built locally.
Mechanics tests (determinism, hook cleanup) run on either; tests that need
trained weights use real() and skip when it returns None.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch

MODEL = "EleutherAI/pythia-70m"
PYTHIA_70M = dict(vocab_size=50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                  intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048,
                  use_parallel_residual=True)


def copy_claims(src: Path, dst: Path) -> Path:
    """Copy claims/ and questions/ into dst with every claim pinned to MODEL and
    reset to `proposed`. The committed claim files name the step 8 model and carry
    their promotion; tests always start from MODEL and the pre-promotion state."""
    # Only the induction question and its Pythia claims: the chat-model claims need real weights.
    (dst / "claims").mkdir(parents=True)
    (dst / "questions").mkdir()
    shutil.copy(src / "questions" / "q.induction_formation.yaml", dst / "questions")
    for p in (src / "claims").glob("*.yaml"):
        if "EleutherAI/pythia" in p.read_text():
            shutil.copy(p, dst / "claims")
    for p in (dst / "claims").glob("*.yaml"):
        text = re.sub(r"EleutherAI/pythia-[\w.]+", MODEL, p.read_text())
        text = re.sub(r"(?m)^status: \S+", "status: proposed", text)
        p.write_text(re.sub(r"(?m)^  (reproduced_from|basis): .*\n", "", text))
    return dst


def cached_revisions(model: str = MODEL) -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model:
                return {ref for rev in repo.revisions for ref in rev.refs}
    except Exception:
        pass
    return set()


def smallest_cached(model: str = MODEL) -> str | None:
    revs = cached_revisions(model)
    if "main" in revs:
        return "main"
    steps = sorted((int(r[4:]), r) for r in revs if r.startswith("step") and r[4:].isdigit())
    return steps[0][1] if steps else None


def real(revision: str | None = None):
    revision = revision or smallest_cached()
    if revision is None or revision not in cached_revisions():
        return None
    from claimci.backends import CpuBackend
    return CpuBackend(offline=True).load(MODEL, revision)


def synthetic(seed: int = 0, revision: str = "synthetic"):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from claimci.backends import CpuBackend
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(**PYTHIA_70M, attn_implementation="eager")
    return CpuBackend.wrap(GPTNeoXForCausalLM(cfg), MODEL, revision)


def any_handle():
    """(handle, kind): real cached weights if available, else synthetic."""
    h = real()
    return (h, "real") if h is not None else (synthetic(), "synthetic")
