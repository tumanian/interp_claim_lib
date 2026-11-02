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


def test_web_app_warns_while_no_worker_is_alive(tmp_path, monkeypatch):
    from claimci.app import web
    from claimci.store.queue import SqliteQueue
    monkeypatch.delenv("CLAIMCI_DB_URL", raising=False)
    assert web.why_not(tmp_path) == {"ingest": None, "verify": None, "expand": None}   # the live app never blocks
    assert "No worker running" in web.notice(tmp_path)
    SqliteQueue(tmp_path / "ledger" / "queue.sqlite").heartbeat("w1")
    assert web.notice(tmp_path) is None


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


def test_web_app_enqueues_instead_of_running(tmp_path, monkeypatch):
    """POST /api/verify puts a recipe on the queue; /api/jobs lists it; approve/cancel act on it."""
    import json, threading, urllib.request
    from http.server import ThreadingHTTPServer
    from claimci.app import web
    from claimci.run import budget
    monkeypatch.delenv("CLAIMCI_DB_URL", raising=False)
    monkeypatch.setattr(budget, "estimate", lambda root, kind, recipe: {"api_tokens": 1, "gpu_seconds": 1})
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(web, "open_queue", None, raising=False)
    from claimci.store import queue as qmod
    monkeypatch.setattr(qmod, "open_queue", lambda r=".": qmod.SqliteQueue(tmp_path / "q.sqlite"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(root))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    def call(path, body=None, method=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method or ("POST" if body is not None else "GET"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        code, j = call("/api/verify", {"claims": ["mech.prev_token_head"]})
        assert code == 200 and j["status"] == "approved" and j["claims"] == ["mech.prev_token_head"]
        again = call("/api/verify", {"claims": ["mech.prev_token_head"]})[1]
        assert again["job"] == j["job"]                                   # same recipe, same job
        code, v = call("/api/jobs")
        assert code == 200 and [x["id"] for x in v["jobs"]] == [j["job"]] and v["backend"] == "SqliteQueue"
        assert call(f"/api/jobs/{j['job']}/cancel", method="POST")[1]["status"] == "cancelled"
        assert call(f"/api/jobs/{j['job']}")[1]["done"] is True
        assert call("/api/ingest", {"source": ""})[0] == 400
        assert urllib.request.urlopen(base + "/queue").status == 200
    finally:
        httpd.shutdown()


def test_expansions_are_judged_by_effect_against_their_control():
    """An expansion transfers if its interval clears its random control's in the predicted
    direction, whatever the paper's threshold; weaker if under half the source model's effect."""
    from claimci.read import cards
    from claimci.schema.loader import load_repo
    from claimci.store.ledger import Ledger
    root = Path(__file__).resolve().parents[1]
    repo = load_repo(root); led = Ledger.for_repo(repo)
    v = lambda cid: cards.verdict(repo, led, cid)
    assert v("refusal.gemma2b.ablation_jbb.gemma22bit") == "transfers"        # 0.16 vs control 0.97: fails <= 0.1, still transfers
    assert v("refusal.gemma2b.ablation_jbb.qwen354b") == "doesn't transfer"   # 0.85 vs 0.92: intervals overlap
    assert v("refusal.gemma2b.addition_harmless.qwen257binstruct") == "transfers, weaker"
    assert v("refusal.gemma2b.ablation_jbb") == "holds"                       # the source's own model keeps its threshold
    assert v("refusal.gemma2b.ablation_coherence.qwen257binstruct") == "fails"   # no control: the threshold decides
    assert cards.original_of(repo, "dir.qwen354b.evil.chen2025.steer_up") == "dir.qwen257binstruct.evil.chen2025.steer_up"
