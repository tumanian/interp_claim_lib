"""Experiment versions (DESIGN 19): the registry, fingerprints, and keep / bump decisions."""
import shutil

import pytest

from claimci.experiments import registry, versions
from claimci.schema import experiments as lockfile


def test_registry_reads_kinds_from_hints_and_params():
    r = registry()
    assert r["caa.steer"].entrypoint == "claimci.experiments.direction:steer_ab"
    assert "intervention" in r["caa.steer"].tags and "intervention" not in r["caa.direction"].tags
    assert r["caa.steer"].params["block"] == "layer" and r["caa.steer"].params["direction"] == "input"
    assert r["caa.steer"].params["coef"] == "scalar"                      # no marker: a scalar
    assert r["steering.persona_delta"].params["interventions"] == "component"   # declared the old way


def test_versions_yaml_matches_the_code():
    """Fails when code changed without a keep / bump decision: run `claimci versions`."""
    assert versions.drift() == {}


def test_fingerprint_follows_the_files_an_experiment_reaches(tmp_path, monkeypatch):
    shutil.copytree(versions.PKG_ROOT, tmp_path / "claimci", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(versions, "PKG_ROOT", tmp_path / "claimci")
    before = {m: versions.fingerprint(m) for m in ("claimci.experiments.direction", "claimci.experiments.induction")}
    ops = tmp_path / "claimci" / "run" / "operations.py"
    ops.write_text(ops.read_text() + "\n# an edit\n")
    assert versions.fingerprint("claimci.experiments.direction") != before["claimci.experiments.direction"]
    assert versions.fingerprint("claimci.experiments.induction") == before["claimci.experiments.induction"]  # never reaches it


@pytest.fixture
def lock_copy(tmp_path, monkeypatch):
    path = tmp_path / "versions.yaml"
    shutil.copy(lockfile.LOCK_PATH, path)
    monkeypatch.setattr(lockfile, "LOCK_PATH", path)
    return path


def test_keep_and_bump(lock_copy, monkeypatch):
    real = versions.current()
    changed = {eid: {**e, "fingerprint": "0" * 16} if eid in ("caa.steer", "caa.direction") else e for eid, e in real.items()}
    monkeypatch.setattr(versions, "current", lambda: changed)
    assert versions.drift() == {"caa.steer": "code changed", "caa.direction": "code changed"}
    v = lockfile.lock()["caa.steer"]["version"]
    done = versions.write(keep=["caa.direction"], bump=["caa.steer"])
    assert done == {"caa.direction": f"kept at v{lockfile.lock()['caa.direction']['version']}", "caa.steer": f"bumped to v{v + 1}"}
    assert versions.drift() == {} and lockfile.lock()["caa.steer"]["version"] == v + 1


def test_new_experiments_start_at_v1(lock_copy, monkeypatch):
    real = versions.current()
    monkeypatch.setattr(versions, "current", lambda: {**real, "new.one": {**real["caa.steer"], "entrypoint": "x:y"}})
    assert versions.drift() == {"new.one": "new experiment"}
    assert versions.write(new_only=True) == {"new.one": "added at v1"} and versions.drift() == {}
