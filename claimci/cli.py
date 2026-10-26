"""claimci command line."""
from __future__ import annotations

import argparse
import sys
import time

from .ledger import Ledger
from .loader import load_repo, write_promotion
from .planner import make_plan
from .validate import validate


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _load_valid(root):
    repo = load_repo(root)
    errors = validate(repo)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return repo, errors


def cmd_validate(args) -> int:
    repo, errors = _load_valid(args.root)
    if errors:
        return 1
    print(f"ok: {len(repo.claims)} claims, {len(repo.questions)} questions")
    return 0


def cmd_plan(args) -> int:
    repo, errors = _load_valid(args.root)
    if errors:
        return 1
    plan = make_plan(repo, Ledger.for_repo(repo), _csv(args.revisions), _csv(args.claims),
                     args.mode, args.force)
    print(plan.render())
    return 0


def verify(repo, backend, revisions=None, claims=None, mode="shared", force=False,
           claim_timeout=600, mem_budget=None, promote=False, log=print):
    from .runner import Runner
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
        for r in results:
            c = repo.claims[r.claim]
            if r.status == "ok" and r.reproduction and c.status == "proposed":
                code = repo.evidence[c.evidence]
                earned = c.evidence if code.rerunnable else c.provenance["source"].get("stated_evidence")
                state = write_promotion(c, r.reproduction["within_tolerance"], earned)
                log(f"{r.claim}: claim file set to {state} ({c.path}); commit it separately")
    return plan, results


def cmd_verify(args) -> int:
    repo, errors = _load_valid(args.root)
    if errors:
        return 1
    from .backends import BACKENDS
    backend = BACKENDS[args.backend](threads=args.threads, offline=args.offline)
    t0 = time.perf_counter()
    try:
        _, results = verify(repo, backend, _csv(args.revisions), _csv(args.claims), args.mode, args.force,
                            args.claim_timeout, args.mem_budget and int(args.mem_budget * 2**20), args.promote)
    except (OSError, MemoryError) as e:  # missing checkpoint (--offline / no network), cache over budget
        print(f"error: {type(e).__name__}: {str(e).splitlines()[0]}", file=sys.stderr)
        return 2
    print(f"total wall clock: {time.perf_counter() - t0:.1f}s")
    return 1 if any(r.status in ("error", "hook_leak") for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claimci")
    p.add_argument("--root", default=".", help="repo root containing claims/, questions/, ledger/")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check claims and questions; no model loads")
    for name in ("plan", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--revisions", help="comma-separated checkpoints, e.g. main,step1000")
        s.add_argument("--claims", help="comma-separated claim ids (ancestors are included)")
        s.add_argument("--mode", choices=["shared", "naive"], default="shared")
        s.add_argument("--force", action="store_true", help="ignore ledger hits")
    v = sub.choices["verify"]
    v.add_argument("--backend", choices=["cpu"], default="cpu")
    v.add_argument("--threads", type=int, default=None, help="default: all cores")
    v.add_argument("--offline", action="store_true", help="fail if a checkpoint is not cached")
    v.add_argument("--claim-timeout", type=float, default=600, help="seconds per claim")
    v.add_argument("--mem-budget", type=float, default=None, help="MB for the shared activation cache")
    v.add_argument("--promote", action="store_true",
                   help="write evidence/status/reproduced_from to proposed claim files that reproduce")
    args = p.parse_args(argv)
    return {"validate": cmd_validate, "plan": cmd_plan, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
