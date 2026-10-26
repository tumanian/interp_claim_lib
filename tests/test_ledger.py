import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from claimci.ledger import Ledger
from claimci.loader import load_repo
from claimci.spec import Result

ROOT = Path(__file__).resolve().parents[1]
IDS = ["causal.induction_head_ablation", "mech.induction_head", "mech.prev_token_head"]


@pytest.fixture
def repo_copy(tmp_path):
    for d in ("claims", "questions"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def row(repo, cid, value=0.5, **kw):
    # Synthetic rows: these tests exercise ledger bookkeeping, not model results.
    c = repo.claims[cid]
    kw.setdefault("status", "ok")
    return Result(claim=cid, claim_key=repo.claim_key(cid), value=value, holds=True,
                  bindings={"layer": 1, "head": 2}, model=c.model, revision=c.revision,
                  code_version=c.code_version, data_hash=c.dataset.hash, dataset=c.dataset.short,
                  lineage=repo.lineage(cid), reproduction={"target": 0.4, "within_tolerance": True},
                  cost={"forward_passes": 1.0, "seconds": 3.2, "backend": "cpu", "mode": "shared"},
                  run={"runner": "claimci 0.1.0", "host": "h", "at": "2026-10-26T21:00Z", "run_id": "r1"},
                  **kw)


def full_run(repo, ledger):
    for cid in IDS:
        ledger.put(row(repo, cid))


def test_put_get_roundtrip(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    r = row(repo, "mech.induction_head", value=0.123456)
    led.put(r)
    assert asdict(led.get(r.claim, r.claim_key)) == asdict(r)
    assert asdict(Ledger.for_repo(repo).get(r.claim, r.claim_key)) == asdict(r)


def test_reopen_yields_same_rows(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    full_run(repo, led)
    assert len(led.rows) == 3
    again = Ledger.for_repo(repo)
    assert [asdict(r) for r in again.rows] == [asdict(r) for r in led.rows]


def test_error_rows_do_not_count_as_results(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    led.put(row(repo, "mech.prev_token_head", status="error", value=None))
    assert led.get("mech.prev_token_head", repo.claim_key("mech.prev_token_head")) is None
    assert "mech.prev_token_head" in led.stale(repo)


def test_stale_lifecycle(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    assert led.stale(repo) == IDS
    assert {led.stale_reason(repo, c) for c in IDS} == {"never run"}
    full_run(repo, led)
    assert led.stale(repo) == []

    p = repo_copy / "claims" / "mech.induction_head.yaml"
    d = yaml.safe_load(p.read_text())
    d["code"]["params"]["min_layer"] = 2
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    edited = load_repo(repo_copy)
    stale = Ledger.for_repo(edited).stale(edited)
    assert stale == ["causal.induction_head_ablation", "mech.induction_head"]
    assert [led.stale_reason(edited, c) for c in stale] == ["code changed", "code changed"]


def test_stale_per_revision(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    full_run(repo, led)
    assert led.stale(repo, "main") == []
    assert led.stale(repo, "step1000") == IDS


def test_stale_reason_model_changed(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    full_run(repo, led)
    p = repo_copy / "claims" / "mech.prev_token_head.yaml"
    p.write_text(p.read_text().replace("pythia-70m", "pythia-160m"))
    edited = load_repo(repo_copy)
    assert "mech.prev_token_head" in led.stale(edited)
    assert led.stale_reason(edited, "mech.prev_token_head") == "model changed"


def test_malformed_ledger_row_names_the_line(repo_copy):
    repo = load_repo(repo_copy)
    led = Ledger.for_repo(repo)
    led.put(row(repo, "mech.prev_token_head"))
    with led.path.open("a") as f:
        f.write('{"claim": "x"}\n')
    with pytest.raises(ValueError, match="results.jsonl:2: malformed ledger row"):
        Ledger.for_repo(repo)
