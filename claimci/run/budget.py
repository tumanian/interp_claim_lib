"""The safety buffer (DESIGN 23): what a job will cost, what it did cost, and whether it may run.

Costs are counted in two currencies: Claude API tokens (input + output, from each response's
usage) and GPU seconds (wall time of a verification). `.claimci-budget.json` at the repo root
caps both per day and per job; under the per-job cap a job is approved as it is enqueued, over it
the job waits for a person. The worker starts nothing once a daily cap is spent.

Accounting is a module-level meter: the worker opens one per job, and the Claude call sites
(sources/extract.py, experiments/judge.py) report every response's usage to it.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {"daily": {"api_tokens": 2_000_000, "gpu_seconds": 4 * 3600},
            "per_job": {"api_tokens": 300_000, "gpu_seconds": 3600},
            "max_open_jobs": 20}
CURRENCIES = ("api_tokens", "gpu_seconds")

# rough per-call sizes, from the runs so far: extraction reads a whole paper twice; a judge call is short
INGEST_TOKENS = 90_000
JUDGE_CALL_TOKENS = 700
JUDGED_ANSWERS_PER_CLAIM = 64


def load(root: Path | str = ".") -> dict:
    p = Path(root) / ".claimci-budget.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    return {"daily": {**DEFAULTS["daily"], **cfg.get("daily", {})},
            "per_job": {**DEFAULTS["per_job"], **cfg.get("per_job", {})},
            "max_open_jobs": cfg.get("max_open_jobs", DEFAULTS["max_open_jobs"])}


# --- the meter ---------------------------------------------------------------------------------

class Meter:
    def __init__(self):
        self.api_tokens = 0
        self.api_calls = 0
        self.gpu_seconds = 0.0
        self._lock = threading.Lock()

    def as_dict(self) -> dict:
        return {"api_tokens": self.api_tokens, "api_calls": self.api_calls, "gpu_seconds": round(self.gpu_seconds, 1)}


_current: Meter | None = None


@contextmanager
def metered():
    """Everything recorded inside lands on the yielded meter (one job at a time per process)."""
    global _current
    m = Meter()
    _current = m
    try:
        yield m
    finally:
        _current = None


def record_api(usage) -> None:
    """Called with a response's `usage` by every Claude call site; a no-op outside a job."""
    if _current is None or usage is None:
        return
    with _current._lock:
        _current.api_calls += 1
        _current.api_tokens += int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)


def record_gpu(seconds: float) -> None:
    if _current is not None:
        _current.gpu_seconds += seconds


# --- estimates ------------------------------------------------------------------------------------

def estimate(root: Path | str, kind: str, recipe: dict) -> dict:
    """A rough cost for a recipe before it runs, in the budget's currencies, with how it was made."""
    root = Path(root)
    if kind == "ingest":
        return {"api_tokens": INGEST_TOKENS, "gpu_seconds": 0, "basis": "two extraction calls over the paper"}
    from ..schema.loader import load_repo
    from ..store.ledger import Ledger
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)
    if kind == "verify":
        ids = recipe.get("claims") or []
        return {**_verify_cost(repo, ledger, ids, bool(recipe.get("force"))), "basis": "planner, from this model's cost history"}
    if kind == "expand":
        from ..sources.expand import closures
        originals = [k for _, ids in closures(repo, recipe.get("source", "")) for k in ids]
        one = _verify_cost(repo, ledger, originals, force=True)
        n = len(recipe.get("models") or [])
        return {"api_tokens": one["api_tokens"] * n, "gpu_seconds": one["gpu_seconds"] * n,
                "basis": f"the source's own claims x {n} models"}
    raise ValueError(kind)


def _verify_cost(repo, ledger, ids: list[str], force: bool) -> dict:
    from .planner import make_plan
    plan = make_plan(repo, ledger, None, ids or None, "shared", force)
    gpu = (plan.sec_per_pass or 60.0) * plan.forward_passes
    to_run = {c for g in plan.groups for c in g.claims}
    judged = sum(1 for c in to_run if (repo.claims[c].experiment_id or "").split(".")[0] in ("judge", "persona"))
    return {"api_tokens": judged * JUDGED_ANSWERS_PER_CLAIM * 2 * JUDGE_CALL_TOKENS, "gpu_seconds": round(gpu)}


# --- decisions ----------------------------------------------------------------------------------

def spent_today(jobs) -> dict:
    """Costs of jobs that finished today (UTC), from the queue."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = {c: 0 for c in CURRENCIES}
    for j in jobs:
        if j.status in ("done", "failed") and (j.finished or "").startswith(today):
            for c in CURRENCIES:
                out[c] += j.cost.get(c, 0) or 0
    return out


def over(cap: dict, amount: dict) -> str | None:
    """Which currency `amount` exceeds `cap` in, or None."""
    for c in CURRENCIES:
        if amount.get(c, 0) > cap.get(c, float("inf")):
            return c
    return None


def admit(budget: dict, est: dict, open_jobs: int) -> tuple[bool, str]:
    """At enqueue: auto-approve under the per-job cap, else hold for a person."""
    if open_jobs >= budget["max_open_jobs"]:
        return False, f"queue has {open_jobs} open jobs (max {budget['max_open_jobs']})"
    c = over(budget["per_job"], est)
    if c:
        return False, f"estimated {est[c]} {c} is over the per-job cap of {budget['per_job'][c]}; needs approval"
    return True, "under the per-job cap"


def may_start(budget: dict, est: dict, spent: dict) -> tuple[bool, str]:
    """At the worker: refuse once today's spend plus this job would pass a daily cap."""
    for c in CURRENCIES:
        if spent.get(c, 0) + est.get(c, 0) > budget["daily"][c]:
            return False, f"daily {c} cap: {spent.get(c, 0)} spent + {est.get(c, 0)} estimated > {budget['daily'][c]}"
    return True, "within today's budget"
