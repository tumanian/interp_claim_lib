"""search / get / reverify on the synthetic induction repo (random-init Pythia, no downloads)."""
import yaml

from claimci import agent_api
from claimci.ledger import Ledger
from claimci.loader import load_repo
from tests.test_runner import copy_repo, run, set_expect

IDS = ["mech.prev_token_head", "mech.induction_head", "causal.induction_head_ablation"]


def test_search_ranks_id_matches_first(tmp_path):
    root = copy_repo(tmp_path)
    repo = load_repo(root)
    out = agent_api.search(repo, Ledger.for_repo(repo), "induction head ablation")
    assert out[0]["id"] == "causal.induction_head_ablation"
    assert {r["id"] for r in out} == set(IDS)
    assert agent_api.search(repo, Ledger.for_repo(repo), "zebra") == []
    assert agent_api.search(repo, Ledger.for_repo(repo), "induction", model="gemma") == []


def test_get_before_and_after_a_run(tmp_path):
    root = copy_repo(tmp_path)
    repo = load_repo(root)
    g = agent_api.get(repo, Ledger.for_repo(repo), "mech.induction_head")
    assert g["stale"] and g["stale_reason"] == "never run" and g["value"] is None and g["status"] == "not run"
    run(root)
    repo = load_repo(root)
    g = agent_api.get(repo, Ledger.for_repo(repo), "mech.induction_head")
    assert not g["stale"] and g["verdict"] == "holds" and g["margin"] == round(g["value"] - g["threshold"]["value"], 6)
    assert g["evidence"] == ("Correlational", "Activation reading")


def test_reverify_puts_previously_verified_claims_first(tmp_path):
    root = copy_repo(tmp_path)
    repo = load_repo(root)
    led = Ledger.for_repo(repo)
    assert [r["why"] for r in agent_api.reverify_plan(repo, led)] == ["never run"] * 3
    run(root)
    repo = load_repo(root)
    assert agent_api.reverify_plan(repo, Ledger.for_repo(repo)) == []
    p = root / "claims" / "mech.prev_token_head.yaml"          # edit one claim's code params: it and its dependents go stale
    d = yaml.safe_load(p.read_text()); d["code"]["params"] = {"unused": 1}; p.write_text(yaml.safe_dump(d, sort_keys=False))
    repo = load_repo(root)
    plan = agent_api.reverify_plan(repo, Ledger.for_repo(repo))
    assert {r["id"] for r in plan} == set(IDS) and all(r["why"] == "code changed" for r in plan)
    assert agent_api.reverify_plan(repo, Ledger.for_repo(repo), skip_models=("pythia",)) == []
