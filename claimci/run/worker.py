"""The worker: claims jobs from the queue and runs them under the budget (DESIGN 23).

    claimci worker [--once] [--poll 5]

One job at a time, oldest approved first. Each job runs with a meter open, its log streamed to
ledger/jobs/<id>.log, and its cost (API tokens, GPU seconds) recorded on the queue. A heartbeat
every poll; a job whose worker went quiet for ten minutes is re-queued once by whoever is alive.
"""
from __future__ import annotations

import os
import signal
import socket
import time
import traceback
from pathlib import Path

from ..store.queue import Job, Queue
from . import budget

STALE_AFTER = 600


def run_job(root: Path, job: Job, log) -> dict:
    """Run one recipe with the existing pipelines; returns what the pipeline returned."""
    r = job.recipe
    if job.kind == "ingest":
        from ..sources.pipeline import ingest_source
        return ingest_source(root, r["source"], r.get("text"), log=log)
    from .executors import executor_for
    t0 = time.perf_counter()
    try:
        if job.kind == "verify":
            rc = executor_for(root).verify(r["claims"], log=log, force=bool(r.get("force")))
            return {"verify_exit": rc}
        if job.kind == "expand":
            from ..sources.expand import expand
            return expand(root, r["source"], r["models"], verify=True, log=log)
    finally:
        budget.record_gpu(time.perf_counter() - t0)
    raise ValueError(f"unknown job kind {job.kind}")


class Worker:
    def __init__(self, root: Path | str, queue: Queue, poll: float = 5.0, log=print):
        self.root, self.queue, self.poll, self.log = Path(root), queue, poll, log
        self.id = f"{socket.gethostname()}-{os.getpid()}"
        self.budget = budget.load(self.root)
        self.stop = False
        (self.root / "ledger" / "jobs").mkdir(parents=True, exist_ok=True)

    def run(self, once: bool = False) -> int:
        """Poll until stopped; with `once`, run at most one job and return."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: setattr(self, "stop", True))
            except ValueError:   # not the main thread
                pass
        ran = 0
        while not self.stop:
            self.queue.heartbeat(self.id)
            for j in self.queue.requeue_stale(STALE_AFTER):
                self.log(f"job {j.id}: worker went quiet, now {j.status}")
            job = self.next_job()
            if job is not None:
                self.run_one(job)
                ran += 1
                if once:
                    break
            elif once:
                break
            else:
                time.sleep(self.poll)
        return ran

    def next_job(self) -> Job | None:
        """The oldest approved job that fits today's budget, claimed; None if nothing can start."""
        spent = budget.spent_today(self.queue.list(limit=500))
        for j in sorted(self.queue.list("approved"), key=lambda j: j.created):
            ok, why = budget.may_start(self.budget, j.estimate, spent)
            if not ok:
                self.log(f"job {j.id} waits: {why}")
                return None   # oldest first: nothing behind it jumps the queue
            claimed = self.queue.claim(self.id)
            if claimed is not None:
                return claimed
        return None

    def run_one(self, job: Job) -> Job:
        path = self.root / "ledger" / "jobs" / f"{job.id}.log"
        self.log(f"job {job.id} ({job.kind}): start")
        with path.open("a") as f, budget.metered() as meter:
            def log(line):
                f.write(str(line) + "\n"); f.flush()
                self.queue.heartbeat(self.id, job.id)
            try:
                out = run_job(self.root, job, log)
                failed = isinstance(out, dict) and (out.get("verify_exit") or out.get("errors"))
                status, error = ("failed", f"pipeline reported {out}"[:500]) if failed else ("done", None)
            except Exception as e:
                log(traceback.format_exc())
                status, error = "failed", f"{type(e).__name__}: {e}"[:500]
        done = self.queue.finish(job.id, status, cost=meter.as_dict(), error=error, log=str(path.relative_to(self.root)))
        self.log(f"job {job.id}: {status}  cost {meter.as_dict()}" + (f"  {error}" if error else ""))
        return done
