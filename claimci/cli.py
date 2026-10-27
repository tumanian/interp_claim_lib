"""claimci command line."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .store.ledger import Ledger
from .schema.loader import load_repo
from .run.planner import make_plan
from .run.verify import verify
from .checks import validate


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _load_valid(root):
    repo = load_repo(root)
    errors = validate(repo)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return repo, errors


def cmd_validate(args) -> int:
    from .experiments.versions import drift
    repo, errors = _load_valid(args.root)
    for eid, what in sorted(drift().items()):
        print(f"error: experiment {eid}: {what}; decide with `claimci versions --keep {eid}` (same computation) "
              f"or `--bump {eid}` (different)", file=sys.stderr)
        errors.append(eid)
    if errors:
        return 1
    print(f"ok: {len(repo.claims)} claims, {len(repo.questions)} questions")
    return 0


def cmd_versions(args) -> int:
    """Show experiments whose code changed since versions.yaml; record keep / bump decisions."""
    from .experiments.versions import drift, write
    if args.keep or args.bump or args.add:
        for eid, what in sorted(write(keep=_csv(args.keep) or (), bump=_csv(args.bump) or (), new_only=args.add).items()):
            print(f"{eid}: {what}")
    pending = drift()
    for eid, what in sorted(pending.items()):
        print(f"{eid}: {what}")
    if not pending:
        print("versions.yaml is up to date")
    return 1 if pending else 0


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


def cmd_verify(args) -> int:
    repo, errors = _load_valid(args.root)
    if errors:
        return 1
    from .run.backends import BACKENDS
    backend = BACKENDS[args.backend](threads=args.threads, offline=args.offline, device=args.device)
    t0 = time.perf_counter()
    try:
        _, results = verify(repo, backend, _csv(args.revisions), _csv(args.claims), args.mode, args.force,
                            args.claim_timeout, args.mem_budget and int(args.mem_budget * 2**20), args.promote)
    except (OSError, MemoryError, KeyError) as e:  # missing checkpoint, cache over budget, unknown claim
        print(f"error: {type(e).__name__}: {str(e).splitlines()[0]}", file=sys.stderr)
        return 2
    print(f"total wall clock: {time.perf_counter() - t0:.1f}s")
    return 1 if any(r.status in ("error", "contaminated") for r in results) else 0


def _load_or_fail(root):
    """Reports and status work on an unvalidated repo, but structural errors still end the command."""
    from .schema.claim import SpecError
    repo = load_repo(root)
    try:
        for cid in repo.claims:
            if not repo.claims[cid].is_source:
                repo.claim_key(cid)
    except SpecError as e:
        raise SystemExit(f"error: {e}")
    for e in repo.errors:
        print(f"warning: {e}", file=sys.stderr)
    return repo


def cmd_matrix(args) -> int:
    from .read.report import matrix_markdown, matrix_png
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
    from .read.report import throughput_markdown
    print(throughput_markdown(Ledger.for_repo(_load_or_fail(args.root))))
    return 0


def cmd_status(args) -> int:
    from .read.report import status_report
    repo = _load_or_fail(args.root)
    print(status_report(repo, Ledger.for_repo(repo), _csv(args.revisions)))
    return 0


def cmd_ingest(args) -> int:
    from pathlib import Path
    from .sources.pipeline import ingest_source
    text = Path(args.text_file).read_text() if args.text_file else None  # posts: text given, URL is the ref
    out = ingest_source(Path(args.root), args.source, text, args.question, do_match=not args.no_match)
    for c in out["claims"]:
        what = (f"same as {c['same_as']}" if c["status"] == "same_as" else
                f"{len(c['tests'])} tests from template {c['template']}" if c.get("tests") else f"gap: {c['gap']}")
        print(f"  {c['id']:<55} quote {'ok' if c['quote_found'] else 'NOT FOUND'}  {what}")
    return 1 if out["errors"] else 0


def cmd_match(args) -> int:
    from pathlib import Path
    from .sources.pipeline import match_source
    out = match_source(Path(args.root), args.ref)
    for c in out["claims"]:
        what = (f"same as {c['same_as']}" if c["same_as"] else
                f"{len(c['tests'])} tests from {c['template']} {c['params']}" if c["tests"] else f"gap: {c['gap']}")
        print(f"  {c['id']:<58} {what}")
    return 1 if out["errors"] else 0


def cmd_expand(args) -> int:
    from pathlib import Path
    from .sources.expand import expand
    out = expand(Path(args.root), args.ref, _csv(args.models), verify=args.verify)
    print(f"wrote {len(out['written'])} claims, refused {len(out['refused'])} closures"
          + (f"; validation errors: {out['errors']}" if out["errors"] else ""))
    return 1 if out["errors"] or out.get("verify_exit") else 0


def cmd_agent(args) -> int:
    from .read import agent as agent_api
    repo = _load_or_fail(args.root)
    ledger = Ledger.for_repo(repo)
    if args.cmd == "search":
        out = agent_api.search(repo, ledger, args.query, args.kind, args.model, args.limit)
        if args.json:
            print(json.dumps(out, indent=1))
        else:
            for r in out:
                print(f"{r['id']:<62} {r['kind']:<12} {r['verdict']}\n    {r['statement'][:150]}")
        return 0
    if args.cmd == "get":
        if args.claim not in repo.claims:
            print(f"error: unknown claim {args.claim}", file=sys.stderr)
            return 1
        print(json.dumps(agent_api.get(repo, ledger, args.claim, args.revision), indent=1, default=str))
        return 0
    plan = agent_api.reverify_plan(repo, ledger, tuple(_csv(args.skip_models) or ()))[:args.limit]
    for r in plan:
        print(f"  {r['id']:<62} {r['why']:<22} ~{r['est_seconds']:.0f}s  {r['model']}")
    print(f"{len(plan)} stale claims" + (" (dry run)" if args.dry_run else ""))
    if args.dry_run or not plan:
        return 0
    from pathlib import Path
    from .run.executors import executor_for
    return executor_for(Path(args.root)).verify([r["id"] for r in plan])


def cmd_card(args) -> int:
    from .read import cards
    repo = _load_or_fail(args.root)
    ledger = Ledger.for_repo(repo)
    if args.cmd == "card":
        if args.claim not in repo.claims:
            print(f"error: unknown claim {args.claim}", file=sys.stderr)
            return 1
        out = [cards.card(repo, ledger, args.claim)]
    else:
        out = cards.explain(repo, ledger, args.claim)
        if not out:
            print(f"error: no source claims for {args.claim}", file=sys.stderr)
            return 1
    print(json.dumps(out if args.cmd == "explain" else out[0], indent=1, default=str) if args.json
          else "\n\n".join(cards.render(d) for d in out))
    return 0


def load_env(root) -> None:
    """KEY=value lines from <root>/.env into the environment; variables already set win."""
    import os
    from pathlib import Path
    path = Path(root) / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.split(" #")[0].strip().strip("'\"")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


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
    v.add_argument("--backend", choices=["torch", "cpu"], default="torch", help="cpu is the v0 name for torch")
    v.add_argument("--device", help="cpu, cuda, mps; default: chat models on CUDA when available, token models on CPU")
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
    ig = sub.add_parser("ingest", help="extract source claims from a paper (arXiv id) or a post (URL + text)")
    ig.add_argument("source", help="arXiv id or URL; for a post, its URL")
    ig.add_argument("--text-file", help="the post's text, when the page cannot be fetched (e.g. X)")
    ig.add_argument("--question", help="attach the claims to this question id")
    ig.add_argument("--prefix", help="claim id prefix, default <first author surname><year>")
    ig.add_argument("--no-match", action="store_true", help="skip matching against existing claims and templates")
    for name, what in (("card", "one claim: threshold and who set it, results, margins, controls"),
                       ("explain", "every source claim from a source (arXiv id, URL, or claim id) and its support")):
        s = sub.add_parser(name, help=what)
        s.add_argument("claim")
        s.add_argument("--json", action="store_true")
    mt = sub.add_parser("match", help="match an ingested source's claims to test templates (no re-extraction)")
    mt.add_argument("ref", help="the source's ref, e.g. arXiv:2507.21509")
    ex = sub.add_parser("expand", help="instantiate a source's verifications on other models")
    ex.add_argument("ref")
    ex.add_argument("--models", required=True, help="comma-separated Hugging Face model ids")
    ex.add_argument("--verify", action="store_true", help="run the new claims (on the GPU box when configured)")
    se = sub.add_parser("search", help="find claims by words in their id, statement, quote or source")
    se.add_argument("query")
    se.add_argument("--kind", choices=["source", "test", "measurement"])
    se.add_argument("--model", help="substring of the model id")
    se.add_argument("--limit", type=int, default=10)
    se.add_argument("--json", action="store_true")
    ge = sub.add_parser("get", help="one claim's current result as JSON (verdict, value, margin, controls, caveats)")
    ge.add_argument("claim")
    ge.add_argument("--revision")
    rv = sub.add_parser("reverify", help="run every stale claim, previously-verified first, cheapest first")
    rv.add_argument("--dry-run", action="store_true")
    rv.add_argument("--limit", type=int, default=1000)
    rv.add_argument("--skip-models", help="comma-separated substrings of model ids to skip (e.g. gated ones)")
    jg = sub.add_parser("judge", help="score question/answer pairs (JSONL) with an LLM judge rubric")
    jg.add_argument("path", help="JSONL with `question` and `answer` per line")
    jg.add_argument("--rubric", default="em")
    jg.add_argument("--model", default="claude-haiku-5-5")
    jg.add_argument("--effort", default="low")
    jg.add_argument("--out", help="default: <input>.judged.jsonl")
    vs = sub.add_parser("versions", help="experiments whose code changed; keep or bump their versions")
    vs.add_argument("--keep", help="comma-separated ids: same computation, keep the version")
    vs.add_argument("--bump", help="comma-separated ids: different computation, next version")
    vs.add_argument("--add", action="store_true", help="add new experiments at v1, decide nothing else")
    sv = sub.add_parser("serve", help="the local web app: ingest sources, read cards, run verification")
    sv.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    load_env(args.root)
    if args.cmd == "judge":
        from .read.judging import main as judge_main
        return judge_main(args.path, args.rubric, args.model, args.effort, args.out)
    if args.cmd == "serve":
        from pathlib import Path
        from .app.web import serve
        serve(Path(args.root), args.port)
        return 0
    return {"validate": cmd_validate, "plan": cmd_plan, "verify": cmd_verify, "matrix": cmd_matrix,
            "throughput": cmd_throughput, "status": cmd_status, "ingest": cmd_ingest,
            "card": cmd_card, "explain": cmd_card,
            "match": cmd_match, "expand": cmd_expand,
            "search": cmd_agent, "get": cmd_agent, "reverify": cmd_agent, "versions": cmd_versions}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
