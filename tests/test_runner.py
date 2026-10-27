"""Runner tests. Mechanics run on synthetic random-init weights with thresholds
lowered in a temp copy (expect is not part of the claim key); the real-model
checks at the bottom skip until pythia-70m@main is in the HF cache."""
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import yaml

import claimci.experiments.induction as induction
from claimci.backends import CpuBackend
from claimci.cli import verify
from claimci.ledger import Ledger
from claimci.loader import load_repo
from tests.models import cached_revisions, copy_claims, synthetic

ROOT = Path(__file__).resolve().parents[1]
IDS = ["mech.prev_token_head", "mech.induction_head", "causal.induction_head_ablation"]


class SyntheticBackend(CpuBackend):
    """CpuBackend whose load builds random-init pythia-70m architecture; never downloads."""

    def load(self, model, revision):
        return synthetic(seed=zlib.crc32(revision.encode()), revision=revision)


def copy_repo(tmp_path, lower=True):
    copy_claims(ROOT, tmp_path)
    if lower:
        for cid, v in [("mech.prev_token_head", 0.0), ("mech.induction_head", 0.0),
                       ("causal.induction_head_ablation", -100.0)]:
            set_expect(tmp_path, cid, v)
    return tmp_path


def set_expect(root, cid, value):
    p = root / "claims" / f"{cid}.yaml"
    d = yaml.safe_load(p.read_text())
    d["expect"]["value"] = value
    p.write_text(yaml.safe_dump(d, sort_keys=False))


def run(root, **kw):
    repo = load_repo(root)
    return verify(repo, SyntheticBackend(), revisions=["main"], log=lambda s: None, **kw)


def by_id(results):
    return {r.claim: r for r in results}


def test_plan_needs_no_model(tmp_path):
    root = copy_repo(tmp_path, lower=False)
    code = ("import sys; from claimci.cli import main; rc = main(['--root', sys.argv[1], 'plan', "
            "'--revisions', 'main']); assert 'torch' not in sys.modules, 'plan imported torch'; sys.exit(rc)")
    env = {**os.environ, "HF_HOME": str(tmp_path / "empty_hf"), "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True, env=env)
    print(r.stdout)
    assert r.returncode == 0, r.stderr
    assert "3 claims  1 shared + 1 intervention pass" in r.stdout
    assert "skipped (ledger hit): 0   to run: 3   forward passes: 2" in r.stdout


def test_verify_rerun_naive_and_determinism(tmp_path):
    root = copy_repo(tmp_path)
    _, first = run(root)
    rows = by_id(first)
    assert [r.claim for r in first] == IDS
    assert all(r.status == "ok" and r.holds for r in first)
    assert rows["causal.induction_head_ablation"].bindings == rows["mech.induction_head"].bindings
    assert len(Ledger.for_repo(load_repo(root)).rows) == 3

    plan, again = run(root)
    assert again == [] and len(plan.skipped) == 3

    _, naive = run(root, mode="naive")
    _, forced = run(root, force=True)
    for cid in IDS:
        assert round(by_id(naive)[cid].value, 6) == round(rows[cid].value, 6)
        assert by_id(forced)[cid].value == rows[cid].value
    shared_passes = sum(r.cost["forward_passes"] for r in first)
    naive_passes = sum(r.cost["forward_passes"] for r in naive)
    print(f"\nforward passes: shared {shared_passes:g}, naive {naive_passes:g}")
    assert shared_passes == pytest.approx(2)
    assert naive_passes == pytest.approx(4)  # 3 clean passes (one per claim) + 1 intervention


def test_failing_parent_blocks_dependent(tmp_path, monkeypatch):
    root = copy_repo(tmp_path)
    set_expect(root, "mech.induction_head", 0.99)
    calls = []
    real = induction.ablate_head_loss_delta
    monkeypatch.setattr(induction, "ablate_head_loss_delta", lambda *a, **k: calls.append(1) or real(*a, **k))
    _, res = run(root)
    rows = by_id(res)
    assert rows["mech.induction_head"].status == "ok" and rows["mech.induction_head"].holds is False
    causal = rows["causal.induction_head_ablation"]
    print(f"\n{causal.status}: {causal.reason}")
    assert causal.status == "blocked" and "mech.induction_head" in causal.reason
    assert calls == []
    assert sum(r.cost["forward_passes"] for r in res) == pytest.approx(1)


def test_raising_experiment_records_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ablation exploded")
    monkeypatch.setattr(induction, "ablate_head_loss_delta", boom)
    _, res = run(copy_repo(tmp_path))
    rows = by_id(res)
    causal = rows["causal.induction_head_ablation"]
    assert causal.status == "error" and "ablation exploded" in causal.reason and causal.traceback_hash
    assert rows["mech.prev_token_head"].status == rows["mech.induction_head"].status == "ok"


def test_hook_leak_canary_aborts_group(tmp_path, monkeypatch):
    def leaky(acts, ctx, layer, head, ablation="zero"):
        module = ctx.adapter.attn_out_proj(layer)
        module.register_forward_pre_hook(lambda m, args: (args[0] * 0,))  # outside the context manager
        return 0.0
    monkeypatch.setattr(induction, "ablate_head_loss_delta", leaky)
    _, res = run(copy_repo(tmp_path))
    causal = by_id(res)["causal.induction_head_ablation"]
    print(f"\n{causal.status}: {causal.reason}")
    assert causal.status == "hook_leak"


@pytest.mark.skipif("main" not in cached_revisions(),
                    reason="EleutherAI/pythia-70m@main not in HF cache (tests never download)")
def test_real_model_all_three_hold(tmp_path):
    root = copy_repo(tmp_path, lower=False)
    cmd = [sys.executable, "-m", "claimci.cli", "--root", str(root), "verify", "--revisions", "main", "--offline"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    rows = Ledger.for_repo(load_repo(root)).rows
    assert len(rows) == 3 and all(x.status == "ok" and x.holds for x in rows)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert "executed: 0   ledger hits: 3" in r2.stdout


def test_promote_judges_only_the_subject_revision(tmp_path):
    root = copy_repo(tmp_path)
    # Reproduction target far from any value: step1000 result would say unreproduced.
    p = root / "claims" / "causal.induction_head_ablation.yaml"
    d = yaml.safe_load(p.read_text())
    d["provenance"]["source"]["stated_value"] = 100.0
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    repo = load_repo(root)
    # subject revision is main; results arrive for step1000 first, then main
    verify(repo, SyntheticBackend(), revisions=["step1000", "main"], promote=True, log=lambda s: None)
    after = load_repo(root).claims["causal.induction_head_ablation"]
    assert after.status == "unreproduced"
    assert "reproduced_from" not in after.provenance
    # and a reproducing value at main promotes: evidence stays IMP, status verified
    d = yaml.safe_load(p.read_text())
    d["status"] = "proposed"
    main_row = by_id([r for r in Ledger.for_repo(repo).rows if r.revision == "main"])["causal.induction_head_ablation"]
    d["provenance"]["source"]["stated_value"] = main_row.value
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    verify(load_repo(root), SyntheticBackend(), revisions=["main"], promote=True, force=True, log=lambda s: None)
    after = load_repo(root).claims["causal.induction_head_ablation"]
    assert after.status == "verified" and after.evidence == "IMP"
    assert after.provenance["reproduced_from"] == "arXiv:2209.11895"


def test_blocked_group_skips_the_shared_pass(tmp_path):
    root = copy_repo(tmp_path)
    set_expect(root, "mech.prev_token_head", 0.99)   # fails on random weights: both dependents blocked
    _, first = run(root)
    assert [r.status for r in first] == ["ok", "blocked", "blocked"]
    _, again = run(root)   # parent is a ledger hit; dependents are blocked again without a model pass
    assert [r.status for r in again] == ["blocked", "blocked"]
    assert sum(r.cost["forward_passes"] for r in again) == 0


def test_raising_intervention_still_counts_its_pass(tmp_path, monkeypatch):
    real = induction.ablate_head_loss_delta

    def raise_after_run(acts, ctx, layer, head, ablation="zero"):
        real(acts, ctx, layer, head, ablation)
        raise RuntimeError("after the hooked pass")
    monkeypatch.setattr(induction, "ablate_head_loss_delta", raise_after_run)
    _, res = run(copy_repo(tmp_path))
    causal = by_id(res)["causal.induction_head_ablation"]
    assert causal.status == "error"
    assert causal.cost["forward_passes"] == pytest.approx(1 + 1 / 3)


def test_hook_leak_does_not_poison_later_groups(tmp_path, monkeypatch):
    root = copy_repo(tmp_path)
    # second dataset on the same checkpoint => a second group after the leaking one
    d = yaml.safe_load((root / "claims" / "mech.prev_token_head.yaml").read_text())
    from claimci.datasets import content_hash, repeated_random_tokens
    d["id"] = "mech.prev_token_head_seed1"
    d["data"]["dataset"]["params"]["seed"] = 1
    d["data"]["hash"] = content_hash(repeated_random_tokens(**d["data"]["dataset"]["params"]))
    (root / "claims" / "mech.prev_token_head_seed1.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    q = root / "questions" / "q.induction_formation.yaml"
    q.write_text(q.read_text().replace("claims: [", "claims: [mech.prev_token_head_seed1, "))
    clean = by_id(run(root, force=True)[1])["mech.prev_token_head_seed1"].value

    def leaky(acts, ctx, layer, head, ablation="zero"):
        ctx.adapter.attn_out_proj(layer).register_forward_pre_hook(lambda m, args: (args[0] * 0,))
        return 0.0
    monkeypatch.setattr(induction, "ablate_head_loss_delta", leaky)
    res = run(root, force=True)[1]
    rows = by_id(res)
    assert rows["causal.induction_head_ablation"].status == "hook_leak"
    assert [r.claim for r in res].index("mech.prev_token_head_seed1") > [r.claim for r in res].index("causal.induction_head_ablation")
    assert rows["mech.prev_token_head_seed1"].status == "ok"
    assert rows["mech.prev_token_head_seed1"].value == clean
