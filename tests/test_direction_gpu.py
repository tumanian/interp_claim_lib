"""End to end on a real small chat model (Qwen2.5-0.5B-Instruct): CAA and Persona Vectors claims
through `verify`. Runs where the model is in the HF cache (the GPU box); skips elsewhere. The
judge is stubbed, so no API calls are made."""
from pathlib import Path

import pytest
import yaml

from claimci.sources import templates
from claimci.run.backends import CpuBackend
from claimci.run.verify import verify
from claimci.store.ledger import Ledger
from claimci.schema.loader import load_repo
from claimci.read.report import computed_status, current_row

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not cached(), reason=f"{MODEL} not in the HF cache (tests never download)")
SRC = {"id": "src.t.claim", "kind": "source", "evidence": "TAS", "status": "proposed", "statement": "s", "question": "q.t",
       "provenance": {"source": {"kind": "paper", "ref": "arXiv:0000.00000", "locator": "Abstract"}}}


def repo_with(tmp_path, pack, edit=None):
    (tmp_path / "claims").mkdir()
    (tmp_path / "questions").mkdir()
    (tmp_path / "claims" / "src.t.claim.yaml").write_text(yaml.safe_dump(SRC))
    claims = templates.contrastive_direction(SRC, MODEL, 12, pack)   # block 12 of 24
    for c in claims:
        c["subject"]["model"]["batch"] = 8
        if edit:
            edit(c)
    templates.write_claims(claims, tmp_path / "claims", tmp_path / "questions")
    return tmp_path, [c["id"] for c in claims]


def run(root, ids):
    repo = load_repo(root)
    _, res = verify(repo, CpuBackend(), claims=ids, log=lambda s: None)
    led = Ledger.for_repo(load_repo(root))
    return {r.claim.split(".")[-1]: r for r in res}, repo, led


def test_caa_end_to_end(tmp_path):
    def shrink(c):  # a slice of the data keeps the test fast; the code path is the same
        if c["code"]["experiment"] == "caa.direction":
            c["code"]["params"]["n"] = 120
    root, ids = repo_with(tmp_path, "caa/sycophancy", shrink)
    r, repo, led = run(root, ids)
    assert {k: v.status for k, v in r.items()} == {k: "ok" for k in r}, {k: v.reason for k, v in r.items()}
    assert r["direction"].value > 0 and r["direction"].bindings["direction"].startswith("artifact:")
    up, down, rnd = r["steer_up"].value, r["steer_down"].value, r["steer_up_random"].value
    print(f"\nCAA on {MODEL}: up {up:+.3f}  down {down:+.3f}  random {rnd:+.3f}")
    # Whether CAA works on a 0.5B model is a finding, not a property of the code (it barely does).
    # What the code must do: a large push through the same hook visibly changes the answer.
    from claimci.run.interventions import add_vector
    from claimci.run.models import load_lm
    from claimci.run.readouts import ab_matching
    h = load_lm(MODEL, "main", {"chat": True, "dtype": "bfloat16", "template": "hf", "batch": 8})

    class Ctx:
        handle = h
        def count(self, n=1):
            pass
    items = __import__("json").loads((ROOT / "data/caa/sycophancy.json").read_text())["test"][:20]
    from claimci.store.artifacts import Store
    d = Store(root / "ledger" / "artifacts").load(r["direction"].bindings["direction"]).to(__import__("torch").float64)
    base = ab_matching(Ctx(), items)
    pushed = ab_matching(Ctx(), items, add_vector(h, 12, d / d.norm() * 50, 1.0))
    assert (pushed - base).abs().max() > 0.01, "a large push through the hook must change P(match)"
    assert (ab_matching(Ctx(), items) - base).abs().max() < 1e-3, "and the hook is gone afterwards"
    for k in ("steer_up", "steer_down", "steer_up_random"):
        assert 0 <= r[k].bindings["baseline"] <= 1 and 0 <= r[k].bindings["steered"] <= 1
    assert r["steer_up"].bindings["baseline"] == r["steer_down"].bindings["baseline"]   # same unsteered pass
    assert computed_status(repo, current_row(repo, led, ids[0], "main"), led) == "measured"


def test_rerun_is_a_ledger_hit(tmp_path):
    root, ids = repo_with(tmp_path, "caa/corrigible-neutral-HHH",
                          lambda c: c["code"]["params"].update(n=60) if c["code"]["experiment"] == "caa.direction" else None)
    run(root, ids)
    _, res = verify(load_repo(root), CpuBackend(), claims=ids, log=lambda s: None)
    assert res == []


def test_persona_end_to_end_with_stub_judge(tmp_path, monkeypatch):
    import claimci.experiments.judge as judge
    calls = []

    def fake(client, model, effort, template, question, answer):
        calls.append(template[:30])
        if "coherence" in template:
            return 80.0
        return 90.0 if any(w in answer.lower() for w in ("harm", "hurt", "suffer", "evil", "destroy", "pain")) else 10.0
    monkeypatch.setattr(judge, "_judge_one", fake)
    monkeypatch.setattr(judge, "judge", lambda pairs, prompts, model, effort: [
        {"aligned": fake(None, model, effort, prompts["aligned"], q, a), "coherent": fake(None, model, effort, prompts["coherent"], q, a)}
        for q, a in pairs])

    def small(c):
        p = c["code"]["params"]
        if c["code"]["experiment"] == "persona.direction":
            p.update(n_per_pair=1, max_new_tokens=48, threshold=0)   # threshold 0: every coherent pair counts
        else:
            p.update(n_per_question=1, max_new_tokens=48)
    root, ids = repo_with(tmp_path, "persona_vectors/evil", small)
    r, repo, led = run(root, ids)
    assert {k: v.status for k, v in r.items()} == {k: "ok" for k in r}, {k: v.reason for k, v in r.items()}
    assert r["direction"].bindings["pairs_kept"] >= 4
    for k in ("steer_up", "steer_up_random", "steer_down"):
        b = r[k].bindings
        assert 0 <= b["coherent_rate"] <= 1 and b["answers"].startswith("artifact:") and b["n"] == 20
    print(f"\nPersona on {MODEL} (stub judge): up {r['steer_up'].value:+.1f}  random {r['steer_up_random'].value:+.1f}")
    assert calls, "the stub judge was used"


def test_judge_needs_no_key_when_stubbed(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import claimci.experiments.judge as judge
    monkeypatch.setattr(judge, "judge", lambda pairs, prompts, model, effort: [{"aligned": 50.0, "coherent": 50.0}] * len(pairs))
    root, ids = repo_with(tmp_path, "persona_vectors/evil", lambda c: c["code"]["params"].update(
        n_per_pair=1, max_new_tokens=16, threshold=0) if c["code"]["experiment"] == "persona.direction" else None)
    r, _, _ = run(root, ids[:1])
    assert r["direction"].status == "ok"
