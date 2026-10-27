"""Verify: plan against the ledger, run what is missing, optionally promote proposed claims."""
from __future__ import annotations

import time

from ..schema.loader import write_promotion
from ..store.ledger import Ledger, gated_by
from .planner import make_plan
from .runner import Runner


def verify(repo, backend, revisions=None, claims=None, mode="shared", force=False,
           claim_timeout=600, mem_budget=None, promote=False, log=print):
    ledger = Ledger.for_repo(repo)
    plan = make_plan(repo, ledger, revisions, claims, mode, force)
    log(plan.render())
    runner = Runner(repo, ledger, backend, claim_timeout, mem_budget, log)
    t0 = time.perf_counter()
    results = runner.execute(plan)
    passes = sum(r.cost.get("forward_passes", 0) for r in results)
    log(f"executed: {len(results)}   ledger hits: {len(plan.skipped)}   "
        f"forward passes: {round(passes, 6):g}   wall: {time.perf_counter() - t0:.1f}s   run: {runner.run_id}")
    if promote:
        # Promotion is judged at the claim's own subject revision (the paper's checkpoint).
        for r in results:
            c = repo.claims[r.claim]
            rep = c.reproduction(r.value) if r.status == "ok" else None
            if not rep or c.status != "proposed" or r.revision != c.revision:
                continue
            earned = repo.earned_code(r.claim)
            reproduced = rep["within_tolerance"] and not gated_by(repo, runner.ledger, r.claim, r.revision)
            c.status = write_promotion(c, reproduced, earned.id,
                                       rep.get("basis", "stated_value"))
            log(f"{r.claim}: claim file set to {c.status} ({c.path}); commit it separately")
    return plan, results
