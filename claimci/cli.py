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
    try:
        plan = make_plan(repo, Ledger.for_repo(repo), _csv(args.revisions), _csv(args.claims),
                         args.mode, args.force)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 1
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
        # Promotion is judged at the claim's own subject revision (the paper's checkpoint).
        for r in results:
            c = repo.claims[r.claim]
            if r.status != "ok" or not r.reproduction or c.status != "proposed" or r.revision != c.revision:
                continue
            earned = repo.earned_code(r.claim)
            c.status = write_promotion(c, r.reproduction["within_tolerance"], earned.id)
            log(f"{r.claim}: claim file set to {c.status} ({c.path}); commit it separately")
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
    except (OSError, MemoryError, KeyError) as e:  # missing checkpoint, cache over budget, unknown claim
        print(f"error: {type(e).__name__}: {str(e).splitlines()[0]}", file=sys.stderr)
        return 2
    print(f"total wall clock: {time.perf_counter() - t0:.1f}s")
    return 1 if any(r.status in ("error", "hook_leak") for r in results) else 0


def _load_or_fail(root):
    """Reports and status work on an unvalidated repo, but structural errors still end the command."""
    from .spec import SpecError
    repo = load_repo(root)
    try:
        for cid in repo.claims:
            repo.claim_key(cid)
    except SpecError as e:
        raise SystemExit(f"error: {e}")
    for e in repo.errors:
        print(f"warning: {e}", file=sys.stderr)
    return repo


def cmd_matrix(args) -> int:
    from .report import matrix_markdown, matrix_png
    repo = _load_or_fail(args.root)
    if args.question not in repo.questions:
        print(f"error: unknown question {args.question}", file=sys.stderr)
        return 1
    ledger = Ledger.for_repo(repo)
    print(matrix_markdown(repo, ledger, args.question))
    if args.png:
        matrix_png(repo, ledger, args.question, args.png)
        print(f"\nwrote {args.png}")
    return 0


def cmd_throughput(args) -> int:
    from .report import throughput_markdown
    print(throughput_markdown(Ledger.for_repo(_load_or_fail(args.root))))
    return 0


def cmd_status(args) -> int:
    from .report import status_report
    repo = _load_or_fail(args.root)
    print(status_report(repo, Ledger.for_repo(repo), _csv(args.revisions)))
    return 0


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
    m = sub.add_parser("matrix", help="claims x revisions for one question")
    m.add_argument("--question", required=True)
    m.add_argument("--png", help="also render the table to this PNG path")
    sub.add_parser("throughput", help="shared vs naive cost from the ledger")
    st = sub.add_parser("status", help="counts by status and evidence; stale claims and why")
    st.add_argument("--revisions", help="default: each claim's subject revision")
    args = p.parse_args(argv)
    return {"validate": cmd_validate, "plan": cmd_plan, "verify": cmd_verify, "matrix": cmd_matrix,
            "throughput": cmd_throughput, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
