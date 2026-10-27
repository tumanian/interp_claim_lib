import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from claimci.schema.loader import load_repo
from claimci.checks import validate
from tests.models import copy_claims

ROOT = Path(__file__).resolve().parents[1]
IDS = ["mech.prev_token_head", "mech.induction_head", "causal.induction_head_ablation"]


@pytest.fixture
def repo_copy(tmp_path):
    return copy_claims(ROOT, tmp_path)


def edit(root, cid, fn):
    p = root / "claims" / f"{cid}.yaml"
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d, sort_keys=False))


def add_claim(root, d):
    (root / "claims" / f"{d['id']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))


def cli_validate(root):
    return subprocess.run([sys.executable, "-m", "claimci.cli", "--root", str(root), "validate"],
                          capture_output=True, text=True)


def test_three_claims_load():
    repo = load_repo(ROOT)
    assert set(IDS) <= set(repo.claims)
    assert repo.claims["mech.prev_token_head"].evidence == "IOB"
    assert repo.claims["causal.induction_head_ablation"].evidence == "IMP"
    assert repo.claims["mech.induction_head"].exports == ["layer", "head"]
    assert repo.questions["q.induction_formation"].claims == IDS
    assert validate(repo) == []


def test_inherit_resolves_to_parent_data_version():
    repo = load_repo(ROOT)
    child = repo.claims["causal.induction_head_ablation"]
    parent = repo.claims["mech.induction_head"]
    assert child.data == {"inherit": "mech.induction_head"}
    assert child.dataset is not None and child.dataset == parent.dataset
    assert child.dataset.gen == "repeated_random_tokens"


def test_code_version_is_the_experiments_version():
    c = load_repo(ROOT).claims["causal.induction_head_ablation"]
    assert c.experiment_id == "induction.ablate_head" and c.entrypoint.endswith(":ablate_head_loss_delta")
    assert c.code_version_kind == "local" and c.code_version == "v1"
    assert "intervention" in c.tags


def test_unknown_experiment_id_is_an_error(repo_copy):
    edit(repo_copy, "mech.prev_token_head", lambda d: d["code"].update(experiment="nope.nothing"))
    _expect_failure(repo_copy, "mech.prev_token_head", "unknown experiment")


def test_claim_key_stable_across_loads():
    a, b = load_repo(ROOT), load_repo(ROOT)
    for cid in IDS:
        assert a.claim_key(cid) == b.claim_key(cid)


def test_expect_does_not_change_key(repo_copy):
    before = {cid: load_repo(repo_copy).claim_key(cid) for cid in IDS}
    edit(repo_copy, "mech.induction_head", lambda d: d["expect"].update(value=0.99))
    after = load_repo(repo_copy)
    assert after.claims["mech.induction_head"].expect["value"] == 0.99
    assert {cid: after.claim_key(cid) for cid in IDS} == before


def test_statement_and_provenance_do_not_change_key(repo_copy):
    before = load_repo(repo_copy).claim_key("mech.induction_head")
    edit(repo_copy, "mech.induction_head",
         lambda d: d.update(statement="reworded", provenance={"source": {"kind": "human"}}))
    assert load_repo(repo_copy).claim_key("mech.induction_head") == before


def test_params_change_key_and_dependents(repo_copy):
    before = {cid: load_repo(repo_copy).claim_key(cid) for cid in IDS}
    edit(repo_copy, "mech.induction_head", lambda d: d["code"]["params"].update(min_layer=2))
    after = load_repo(repo_copy)
    assert after.claim_key("mech.induction_head") != before["mech.induction_head"]
    assert after.claim_key("causal.induction_head_ablation") != before["causal.induction_head_ablation"]
    assert after.claim_key("mech.prev_token_head") == before["mech.prev_token_head"]


def test_revision_changes_key():
    repo = load_repo(ROOT)
    assert repo.claim_key("mech.induction_head", "step0") != repo.claim_key("mech.induction_head", "main")


def test_cli_validate_ok():
    r = cli_validate(ROOT)
    assert r.returncode == 0, r.stderr


# --- injected errors: validate must exit non-zero and name the claim ---

def _expect_failure(root, claim_id, fragment):
    r = cli_validate(root)
    print(r.stderr)
    assert r.returncode != 0
    assert claim_id in r.stderr and fragment in r.stderr


def test_cycle(repo_copy):
    edit(repo_copy, "mech.prev_token_head",
         lambda d: d.update(depends_on=["causal.induction_head_ablation"]))
    _expect_failure(repo_copy, "mech.prev_token_head", "cycle in depends_on")


def test_input_not_exported(repo_copy):
    edit(repo_copy, "causal.induction_head_ablation",
         lambda d: d["inputs"].update(layer="mech.induction_head.lyr"))
    _expect_failure(repo_copy, "causal.induction_head_ablation", "mech.induction_head.lyr")


def _literature_claim(cid, evidence):
    return {
        "id": cid, "question": "q.induction_formation", "evidence": evidence, "status": "proposed",
        "statement": "x", "subject": {"model": {"hf": "EleutherAI/pythia-70m"}},
        "code": {}, "data": {"inherit": "mech.induction_head"},
        "provenance": {"source": {"kind": "paper", "ref": "arXiv:2209.11895", "locator": "Fig. 1"}},
    }


def test_tas_dependency(repo_copy):
    add_claim(repo_copy, _literature_claim("lit.induction_tas", "TAS"))
    edit(repo_copy, "mech.induction_head", lambda d: d["depends_on"].append("lit.induction_tas"))
    _expect_failure(repo_copy, "mech.induction_head", "lit.induction_tas has evidence TAS")


def test_icp_under_imp(repo_copy):
    probe = _literature_claim("probe.induction_icp", "ICP")
    probe.update(code={"entrypoint": "claimci.experiments.induction:prev_token_score", "version": "local"},
                 metric="probe_acc", expect={"op": ">=", "value": 0.9})
    probe["provenance"] = {}
    add_claim(repo_copy, probe)
    edit(repo_copy, "causal.induction_head_ablation",
         lambda d: d["depends_on"].append("probe.induction_icp"))
    _expect_failure(repo_copy, "causal.induction_head_ablation", "cannot support causal claims")


def test_observational_cannot_be_relabeled_causal(repo_copy):
    edit(repo_copy, "mech.prev_token_head", lambda d: d.update(evidence="IMP"))
    _expect_failure(repo_copy, "mech.prev_token_head", "requires an entrypoint tagged @intervention")


# --- review round: helper edits, lifecycle rules, promotion path, crash cases ---

def test_default_plan_uses_each_claims_own_revision(repo_copy):
    from claimci.store.ledger import Ledger
    from claimci.run.planner import make_plan
    edit(repo_copy, "mech.prev_token_head", lambda d: d["subject"]["model"].update(revision="step1000"))
    repo = load_repo(repo_copy)
    plan = make_plan(repo, Ledger.for_repo(repo))
    assert {(g.revision, tuple(g.claims)) for g in plan.groups} == {
        ("main", ("mech.induction_head", "causal.induction_head_ablation")),
        ("step1000", ("mech.prev_token_head",))}


def test_tas_claim_with_attached_experiment_is_runnable(repo_copy):
    from claimci.store.ledger import Ledger
    from claimci.run.planner import make_plan
    d = _literature_claim("lit.ablation_tas", "TAS")
    d.update(code={"entrypoint": "claimci.experiments.induction:ablate_head_loss_delta", "version": "local",
                   "params": {"ablation": "zero"}},
             metric="loss_delta_nats", expect={"op": ">=", "value": 0.3}, tolerance=0.05,
             depends_on=["mech.induction_head"],
             inputs={"layer": "mech.induction_head.layer", "head": "mech.induction_head.head"})
    d["provenance"]["source"].update(stated_value=0.4, stated_evidence="IMP")
    add_claim(repo_copy, d)
    repo = load_repo(repo_copy)
    assert validate(repo) == []
    assert repo.earned_code("lit.ablation_tas").id == "IMP"
    plan = make_plan(repo, Ledger.for_repo(repo), ["main"])
    assert "lit.ablation_tas" in plan.groups[0].interventional
    # the same claim with an observational stated_evidence contradicts its entrypoint
    edit(repo_copy, "lit.ablation_tas", lambda d: d["provenance"]["source"].update(stated_evidence="IOB"))
    _expect_failure(repo_copy, "lit.ablation_tas", "intervention but evidence IOB")


@pytest.mark.parametrize("cid, change, fragment", [
    ("mech.prev_token_head", lambda d: d["data"]["dataset"]["params"].update(half=-1), "cannot be generated"),
    ("mech.prev_token_head", lambda d: d["data"]["dataset"]["params"].update(vocab_lo=20000), "cannot be generated"),
    ("mech.prev_token_head", lambda d: d["subject"]["model"].pop("hf"), "subject.model.hf"),
    ("mech.prev_token_head", lambda d: d.update(expect=">= 0.5"), "expect must be"),
    ("mech.prev_token_head", lambda d: d["expect"].pop("value"), "expect must be"),
    ("mech.induction_head", lambda d: d.update(depends_on="mech.prev_token_head"), "`depends_on` must be a list"),
    ("causal.induction_head_ablation", lambda d: d["data"].update(hash="deadbeef"), "inherit cannot be combined"),
    ("mech.prev_token_head", lambda d: d.update(status="refuted"), "requires qualifier NOT"),
])
def test_validate_reports_instead_of_crashing(repo_copy, cid, change, fragment):
    edit(repo_copy, cid, change)
    _expect_failure(repo_copy, cid, fragment)


def test_tas_claim_cannot_be_verified(repo_copy):
    d = _literature_claim("lit.induction_tas", "TAS")
    d["status"] = "verified"
    add_claim(repo_copy, d)
    _expect_failure(repo_copy, "lit.induction_tas", "stay proposed")


def test_empty_question_file_is_reported(repo_copy):
    (repo_copy / "questions" / "empty.yaml").write_text("")
    r = cli_validate(repo_copy)
    assert r.returncode != 0 and "empty.yaml" in r.stderr


def test_unstated_expect_must_be_recorded_as_supplemented(repo_copy):
    edit(repo_copy, "mech.induction_head", lambda d: d["provenance"].pop("supplemented"))
    _expect_failure(repo_copy, "mech.induction_head", "provenance.supplemented.expect")


# --- kinds: source / test / measurement ---------------------------------------

def _source(cid, **kw):
    d = {"id": cid, "kind": "finding", "evidence": "NAS", "status": "proposed", "hedge": "shows",
         "statement": "x", "quote": "x", "provenance": {"source": {"kind": "post", "ref": "https://example.org/p"}}}
    d.update(kw)
    return d


def test_source_claim_needs_no_model_data_or_code(repo_copy):
    add_claim(repo_copy, _source("src.post"))
    edit(repo_copy, "mech.induction_head", lambda d: d.update(about="src.post"))
    assert cli_validate(repo_copy).returncode == 0
    assert load_repo(repo_copy).claims["mech.induction_head"].kind == "test"   # inferred from expect


def test_source_claim_cannot_carry_a_threshold(repo_copy):
    add_claim(repo_copy, _source("src.post", expect={"op": ">=", "value": 1}))
    _expect_failure(repo_copy, "src.post", "has no code or expect")


def test_about_must_name_a_source(repo_copy):
    edit(repo_copy, "mech.induction_head", lambda d: d.update(about="mech.prev_token_head"))
    _expect_failure(repo_copy, "mech.induction_head", "about must name a finding")


def test_measurement_has_no_expect(repo_copy):
    edit(repo_copy, "mech.induction_head", lambda d: d.update(kind="measurement"))
    _expect_failure(repo_copy, "mech.induction_head", "a measurement has no `expect`")


def test_expect_holds_and_margin_from_the_threshold_and_an_interval():
    from claimci.schema.claim import Expect
    e = Expect(">=", 0.5)
    assert e.holds(0.6) and e.margin(0.6) == 0.1 and e.holds(0.4) is False and e.margin(0.4) == -0.1
    assert e.holds(0.6, interval=(0.45, 0.75)) is False and e.margin(0.6, interval=(0.45, 0.75)) == -0.05
    assert Expect("<=", 0.2).margin(0.1, interval=(0.05, 0.15)) == 0.05
    near = Expect("==", 0.91, tol=0.01)
    assert near.holds(0.915) and not near.holds(0.93) and near.margin(0.905) == 0.005
    assert Expect.of(None) is None and Expect.of({"op": "<", "value": 1}).holds(0.5)


def test_intervals():
    from claimci.run.stats import bootstrap, wilson
    lo, hi = wilson(60, 64)
    assert 0.84 < lo < 0.86 and 0.97 < hi < 0.99 and wilson(0, 10)[0] == 0.0 and wilson(0, 0) is None
    pairs = [(0.4, 0.6), (0.5, 0.7), (0.45, 0.62), (0.5, 0.71)]
    a = bootstrap(lambda p: (p[:, 1] - p[:, 0]).mean(), pairs)
    assert a == bootstrap(lambda p: (p[:, 1] - p[:, 0]).mean(), pairs) and a[0] <= 0.19 <= a[1]   # seeded


def test_status_lifecycle():
    import pytest
    from claimci.schema.lifecycle import TransitionError, check
    check("proposed", "verified", by="run")
    check("unreproduced", "verified", by="run")
    check("verified", "refuted", by="person")
    with pytest.raises(TransitionError, match="decided by a person"):
        check("proposed", "refuted", by="run")
    with pytest.raises(TransitionError, match="cannot go from verified to proposed"):
        check("verified", "proposed", by="person")


def test_env_file_fills_missing_variables_only(tmp_path, monkeypatch):
    import os
    from claimci.cli import load_env
    (tmp_path / ".env").write_text("# comment\nCLAIMCI_A=from-file   # note\nCLAIMCI_B='quoted'\nCLAIMCI_C=\nCLAIMCI_D=from-file\n")
    monkeypatch.delenv("CLAIMCI_A", raising=False); monkeypatch.delenv("CLAIMCI_B", raising=False)
    monkeypatch.delenv("CLAIMCI_C", raising=False); monkeypatch.setenv("CLAIMCI_D", "from-shell")
    load_env(tmp_path)
    assert os.environ["CLAIMCI_A"] == "from-file" and os.environ["CLAIMCI_B"] == "quoted"
    assert "CLAIMCI_C" not in os.environ and os.environ["CLAIMCI_D"] == "from-shell"
