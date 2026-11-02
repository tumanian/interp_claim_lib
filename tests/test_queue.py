"""The job queue on SQLite: dedupe, the state machine, claiming, heartbeats (no network)."""
import pytest

from claimci.store.queue import SqliteQueue, recipe_hash


@pytest.fixture
def q(tmp_path):
    return SqliteQueue(tmp_path / "queue.sqlite")


def test_same_recipe_is_one_open_job(q):
    a = q.add("verify", {"claims": ["x", "y"]}, requested_by="me")
    b = q.add("verify", {"claims": ["x", "y"]})
    assert a.id == b.id and a.status == "pending" and b.requested_by == "me"
    assert recipe_hash("verify", {"claims": ["x", "y"]}) == a.recipe_hash
    assert q.add("verify", {"claims": ["y", "x"]}).id != a.id       # order is part of the recipe
    q.cancel(a.id)
    assert q.add("verify", {"claims": ["x", "y"]}).id != a.id       # a cancelled job does not block a new one


def test_lifecycle_and_illegal_moves(q):
    j = q.add("ingest", {"source": "arXiv:1"})
    with pytest.raises(ValueError):
        q.finish(j.id, "done")                                       # not running
    assert q.approve(j.id).status == "approved"
    with pytest.raises(ValueError):
        q.approve(j.id)                                              # already approved
    assert q.claim("w1").id == j.id and q.get(j.id).status == "running" and q.get(j.id).worker == "w1"
    assert q.claim("w2") is None                                     # nothing else approved
    with pytest.raises(ValueError):
        q.cancel(j.id)                                               # running jobs are not cancelled
    done = q.finish(j.id, "done", cost={"api_tokens": 1200}, log="artifact:abc")
    assert done.status == "done" and done.cost == {"api_tokens": 1200} and done.finished
    with pytest.raises(ValueError):
        q.add("judge", {})


def test_claim_takes_the_oldest_approved(q):
    first = q.add("verify", {"claims": ["a"]}, approved=True)
    q.add("verify", {"claims": ["b"]})                               # pending: not claimable
    second = q.add("verify", {"claims": ["c"]}, approved=True)
    assert q.claim("w").id == first.id and q.claim("w").id == second.id and q.claim("w") is None
    assert [j.status for j in q.list()] == ["running", "pending", "running"]
    assert [j.id for j in q.list("pending")] != []


def test_dead_worker_requeues_once(q):
    j = q.add("verify", {"claims": ["a"]}, approved=True)
    q.claim("w"); q.heartbeat("w", j.id)
    assert q.requeue_stale(older_than_seconds=3600) == []           # heartbeat is fresh
    assert q.requeue_stale(older_than_seconds=-1)[0].status == "approved"
    q.claim("w2"); q.heartbeat("w2", j.id)
    assert q.requeue_stale(older_than_seconds=-1)[0].status == "failed"
    assert {w["id"] for w in q.workers()} == {"w", "w2"}
