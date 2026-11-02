"""Budget decisions, token accounting and the worker loop, on SQLite with stubbed pipelines."""
from types import SimpleNamespace

import pytest

from claimci.run import budget, worker
from claimci.store.queue import SqliteQueue

ROOT_BUDGET = {"daily": {"api_tokens": 1000, "gpu_seconds": 100}, "per_job": {"api_tokens": 300, "gpu_seconds": 50}, "max_open_jobs": 3}


def test_admit_and_may_start():
    assert budget.admit(ROOT_BUDGET, {"api_tokens": 100, "gpu_seconds": 10}, 0) == (True, "under the per-job cap")
    ok, why = budget.admit(ROOT_BUDGET, {"api_tokens": 500, "gpu_seconds": 10}, 0)
    assert not ok and "per-job cap" in why
    assert not budget.admit(ROOT_BUDGET, {"api_tokens": 1, "gpu_seconds": 1}, 3)[0]       # queue full
    assert budget.may_start(ROOT_BUDGET, {"api_tokens": 100, "gpu_seconds": 0}, {"api_tokens": 950, "gpu_seconds": 0})[0] is False
    assert budget.may_start(ROOT_BUDGET, {"api_tokens": 50, "gpu_seconds": 0}, {"api_tokens": 950, "gpu_seconds": 0})[0] is True


def test_meter_counts_api_usage_only_inside_a_job():
    budget.record_api(SimpleNamespace(input_tokens=5, output_tokens=5))       # outside: ignored
    with budget.metered() as m:
        budget.record_api(SimpleNamespace(input_tokens=100, output_tokens=20))
        budget.record_api(None)
        budget.record_gpu(2.5)
    assert m.as_dict() == {"api_tokens": 120, "api_calls": 1, "gpu_seconds": 2.5}


def test_spent_today_sums_finished_jobs(tmp_path):
    q = SqliteQueue(tmp_path / "q.sqlite")
    a = q.add("verify", {"claims": ["a"]}, approved=True); q.claim("w"); q.finish(a.id, "done", cost={"api_tokens": 10, "gpu_seconds": 3})
    b = q.add("verify", {"claims": ["b"]}, approved=True); q.claim("w"); q.finish(b.id, "failed", cost={"api_tokens": 5})
    q.add("verify", {"claims": ["c"]})                                        # pending: not spent
    assert budget.spent_today(q.list()) == {"api_tokens": 15, "gpu_seconds": 3}


@pytest.fixture
def stub_pipeline(monkeypatch):
    calls = []

    def fake_run_job(root, job, log):
        calls.append(job.id)
        log("ran " + job.kind)
        budget.record_api(SimpleNamespace(input_tokens=40, output_tokens=2))
        if job.recipe.get("claims") == ["boom"]:
            raise RuntimeError("exploded")
        return {"verify_exit": 1 if job.recipe.get("claims") == ["bad"] else 0}
    monkeypatch.setattr(worker, "run_job", fake_run_job)
    return calls


def test_worker_runs_approved_jobs_in_order_and_records_outcomes(tmp_path, stub_pipeline, monkeypatch):
    monkeypatch.setattr(budget, "load", lambda root: ROOT_BUDGET)
    q = SqliteQueue(tmp_path / "q.sqlite")
    ok = q.add("verify", {"claims": ["a"]}, approved=True, estimate={"api_tokens": 10, "gpu_seconds": 0})
    bad = q.add("verify", {"claims": ["bad"]}, approved=True, estimate={"api_tokens": 10, "gpu_seconds": 0})
    boom = q.add("verify", {"claims": ["boom"]}, approved=True, estimate={"api_tokens": 10, "gpu_seconds": 0})
    held = q.add("verify", {"claims": ["held"]}, estimate={"api_tokens": 10, "gpu_seconds": 0})   # pending
    w = worker.Worker(tmp_path, q, poll=0.01, log=lambda m: None)
    assert w.run(once=True) == 1 and q.get(ok.id).status == "done" and q.get(ok.id).cost["api_tokens"] == 42
    assert (tmp_path / q.get(ok.id).log).read_text() == "ran verify\n"
    assert w.run(once=True) == 1 and q.get(bad.id).status == "failed" and "pipeline reported" in q.get(bad.id).error
    assert w.run(once=True) == 1 and q.get(boom.id).status == "failed" and "exploded" in q.get(boom.id).error
    assert w.run(once=True) == 0 and q.get(held.id).status == "pending" and stub_pipeline == [ok.id, bad.id, boom.id]


def test_worker_waits_when_the_daily_cap_is_spent(tmp_path, stub_pipeline, monkeypatch):
    monkeypatch.setattr(budget, "load", lambda root: ROOT_BUDGET)
    q = SqliteQueue(tmp_path / "q.sqlite")
    big = q.add("verify", {"claims": ["x"]}, approved=True, estimate={"api_tokens": 990, "gpu_seconds": 0})
    msgs = []
    w = worker.Worker(tmp_path, q, poll=0.01, log=msgs.append)
    assert w.run(once=True) == 1 and q.get(big.id).status == "done"           # first job fits (0 spent)
    monkeypatch.setattr(w, "budget", {**ROOT_BUDGET, "daily": {"api_tokens": 100, "gpu_seconds": 100}})   # 42 already spent
    small = q.add("verify", {"claims": ["y"]}, approved=True, estimate={"api_tokens": 70, "gpu_seconds": 0})
    assert w.run(once=True) == 0 and q.get(small.id).status == "approved" and any("daily api_tokens cap" in m for m in msgs)
