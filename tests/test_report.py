from pathlib import Path

import yaml

from claimci.store.ledger import Ledger
from claimci.schema.loader import load_repo
from claimci.read.report import matrix, matrix_markdown, matrix_png, status_report, throughput, throughput_markdown
from tests.test_runner import IDS, by_id, copy_repo, run


def test_reports_after_runs(tmp_path):
    root = copy_repo(tmp_path)
    _, shared = run(root)
    run(root, mode="naive")
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)

    md = matrix_markdown(repo, ledger, "q.induction_formation")
    print("\n" + md)
    claims, revs, cells = matrix(repo, ledger, "q.induction_formation")
    assert claims == IDS and revs == ["main"]
    for cid, r in by_id(shared).items():
        assert cells[cid]["main"] == (f"{r.value:.3f} ✓", "pass")
    png = Path(tmp_path / "matrix.png")
    matrix_png(repo, ledger, "q.induction_formation", str(png))
    assert png.stat().st_size > 1000

    print(throughput_markdown(ledger))
    t = {r["mode"]: r for r in throughput(ledger)}
    assert t["shared"]["claims"] == t["naive"]["claims"] == 3
    assert t["shared"]["passes"] < t["naive"]["passes"]

    st = status_report(repo, ledger)
    print(st)
    assert "stale: 0" in st and "verified=3" in st

    p = root / "claims" / "mech.induction_head.yaml"
    d = yaml.safe_load(p.read_text())
    d["code"]["params"]["min_layer"] = 2
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    edited = load_repo(root)
    st = status_report(edited, Ledger.for_repo(edited))
    print(st)
    lines = [ln.split() for ln in st.splitlines() if ln.startswith("  ")]
    assert "stale: 2" in st
    assert {(ln[1], " ".join(ln[2:])) for ln in lines} == {
        ("mech.induction_head", "code changed"), ("causal.induction_head_ablation", "code changed")}


def test_expect_edit_reevaluates_without_rerun(tmp_path):
    root = copy_repo(tmp_path)
    run(root)
    p = root / "claims" / "mech.prev_token_head.yaml"
    d = yaml.safe_load(p.read_text())
    d["expect"]["value"] = 0.99
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)
    _, _, cells = matrix(repo, ledger, "q.induction_formation")
    assert cells["mech.prev_token_head"]["main"][1] == "fail"
    assert ledger.stale(repo) == []
