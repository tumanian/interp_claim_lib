"""search / get / reverify on the synthetic induction repo (random-init Pythia, no downloads)."""
from pathlib import Path

import yaml

from claimci.read import agent as agent_api
from claimci.store.ledger import Ledger
from claimci.schema.loader import load_repo
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
    assert g["evidence"] == "IOB" and g["evidence_label"] == "Correlational | Activation reading" and g["run_on"] == "original"


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
    # the edited claim: its parameters changed; its dependents: an upstream key changed
    assert {r["id"] for r in plan} == set(IDS) and all(r["why"] in ("parameters or inputs changed", "code changed") for r in plan)
    assert all(r["why"] != "never run" for r in plan)
    assert agent_api.reverify_plan(repo, Ledger.for_repo(repo), skip_models=("pythia",)) == []


def test_web_app_says_why_an_action_cannot_run_here(tmp_path, monkeypatch):
    from claimci.app import web
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(web, "_cuda", lambda: False)
    w = web.why_not(tmp_path)
    assert "ANTHROPIC_API_KEY" in w["ingest"] and "GPU" in w["verify"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    (tmp_path / ".claimci-remote.json").write_text("{}")
    assert web.why_not(tmp_path) == {"ingest": None, "verify": None}


def test_static_site_names_every_request_the_page_makes(tmp_path):
    import json
    from claimci.app.static_site import export, key
    root = Path(__file__).resolve().parents[1]
    export(root, tmp_path / "site", base="/demo/")
    page = (tmp_path / "site" / "index.html").read_text()
    assert '<base href="/demo/">' in page and "window.CLAIMCI_STATIC = true" in page
    overview = json.loads((tmp_path / "site" / "api" / key("overview")).read_text())
    assert overview["why_not"]["ingest"] and overview["why_not"]["verify"]          # read-only
    for s in overview["sources"]:
        assert (tmp_path / "site" / "api" / key(f"report?ref={s['ref']}")).exists()


def test_agent_read_path_on_the_committed_results():
    from claimci.read import agent, cards
    from claimci.schema.loader import load_repo
    from claimci.store.ledger import Ledger
    root = Path(__file__).resolve().parents[1]
    repo = load_repo(root); led = Ledger.for_repo(repo)
    hits = agent.search(repo, led, "steering", kind="test", model="qwen", limit=100)
    assert hits and all(h["run_on"] in ("original", "expanded") for h in hits)
    assert {"original", "expanded"} <= {h["run_on"] for h in hits}
    assert all(h["evidence"] == "ISS" for h in agent.search(repo, led, "", kind="test", evidence="iss", limit=100))
    g = agent.get(repo, led, "dir.qwen354b.sycophancy.panickssery2023.steer_up")
    assert g["run_on"] == "expanded" and g["expanded_from"].startswith("meta-llama/") and g["evidence"] == "ISS"
    assert cards.explain(repo, led, "chen2025") == cards.explain(repo, led, "arXiv:2507.21509") != []
