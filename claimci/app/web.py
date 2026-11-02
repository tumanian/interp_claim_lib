"""A small local web app: drop a paper or post, see its claims, verify them.

Standard library only. Ingest calls the Claude API from this machine; verification runs
through an executor: the remote GPU box when .claimci-remote.json exists, else locally (claimci/run/executors.py).

    claimci serve --port 8765
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..read import cards
from ..store.ledger import Ledger
from ..schema.loader import load_repo

PAGE = Path(__file__).with_name("static") / "index.html"
QUEUE_PAGE = Path(__file__).with_name("static") / "queue.html"


def why_not(root: Path) -> dict:
    """{action: the reason it cannot run, or None}. The live app only enqueues, so nothing is
    blocked here; the read-only static export sets its own reasons (static_site.py)."""
    return {"ingest": None, "verify": None, "expand": None}


def notice(root: Path) -> str | None:
    """A warning for the page: the queue needs a worker (with the Claude key and the GPU box)."""
    from ..store.queue import open_queue
    if any(_age(w["seen"]) < 120 for w in open_queue(root).workers()):
        return None
    return "No worker running: jobs will wait in the queue until `claimci worker` starts (it needs ANTHROPIC_API_KEY and the GPU box)."


def overview(root: Path) -> dict:
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)
    sources: dict[str, dict] = {}
    for k, c in sorted(repo.claims.items()):
        if not c.is_finding:
            continue
        src = c.provenance.get("source") or {}
        s = sources.setdefault(src.get("ref", k), {"ref": src.get("ref", k), "kind": src.get("kind"),
                                                   "title": src.get("title") or src.get("ref"), "claims": [],
                                                   "question": c.question})
        s["claims"].append({"id": k, "support": cards.card(repo, ledger, k)["support"]})
    return {"sources": list(sources.values()), "remote": (root / ".claimci-remote.json").exists(),
            "why_not": why_not(root), "notice": notice(root)}


def samples(root: Path, ref: str) -> list[dict]:
    """Saved judged answers (a JSON artifact), with each question's text filled in."""
    import re
    m = re.fullmatch(r"artifact:([0-9a-f]{24})", ref or "")
    if not m:
        raise KeyError("artifact reference")
    rows = json.loads((root / "ledger" / "artifacts" / f"{m.group(1)}.json").read_text())
    from ..store.datasets import DATA_DIR
    qs = {q["id"]: q["question"] for q in
          json.loads((DATA_DIR / "prompts" / "em_first_plot_questions.json").read_text())["questions"]}
    return [{**r, "prompt": r.get("prompt") or qs.get(r.get("question"), r.get("question"))} for r in rows]


def source_claims_to_verify(root: Path, ref: str) -> list[str]:
    """Every test and measurement in the questions of this source's claims."""
    repo = load_repo(root)
    qs = {c.question for c in repo.claims.values() if c.is_finding
          and (c.provenance.get("source") or {}).get("ref") == ref}
    return sorted(k for k, c in repo.claims.items() if not c.is_finding and c.question in qs)


def enqueue(root: Path, kind: str, recipe: dict, requested_by: str | None = None) -> dict:
    """Estimate, admit under the budget, and put the recipe on the queue (DESIGN 23)."""
    from ..run import budget
    from ..store.queue import open_queue
    q = open_queue(root)
    est = budget.estimate(root, kind, recipe)
    ok, why = budget.admit(budget.load(root), est, sum(len(q.list(s)) for s in ("pending", "approved", "running")))
    job = q.add(kind, recipe, requested_by=requested_by, estimate=est, approved=ok)
    return {"job": job.id, "status": job.status, "estimate": est, "admitted": why}


def job_view(root: Path, job_id: str, since: int = 0) -> dict:
    """A job with the lines of its log from `since`: what the page polls."""
    from ..store.queue import open_queue
    job = open_queue(root).get(job_id)
    if job is None:
        raise KeyError(job_id)
    lines = (root / job.log).read_text().splitlines() if job.log and (root / job.log).exists() else []
    return {**job.to_dict(), "done": job.status in ("done", "failed", "cancelled"), "lines": lines[since:], "n": len(lines)}


def queue_view(root: Path) -> dict:
    """Everything the monitoring page shows: jobs, workers, today's spend against the caps."""
    from ..run import budget
    from ..store.queue import open_queue
    q = open_queue(root)
    jobs = q.list(limit=200)
    cfg = budget.load(root)
    workers = q.workers()
    alive = [w for w in workers if _age(w["seen"]) < 120]
    return {"jobs": [j.to_dict() for j in jobs], "workers": workers, "worker_alive": bool(alive),
            "spent_today": budget.spent_today(jobs), "budget": cfg, "backend": type(q).__name__}


def _age(stamp: str) -> float:
    from datetime import datetime, timezone
    try:
        return datetime.now(timezone.utc).timestamp() - datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return float("inf")


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, PAGE.read_text(), "text/html; charset=utf-8")
                if u.path == "/queue":
                    return self.send(200, QUEUE_PAGE.read_text(), "text/html; charset=utf-8")
                if u.path == "/api/jobs":
                    return self.send(200, queue_view(root))
                if u.path == "/api/overview":
                    return self.send(200, overview(root))
                repo = load_repo(root)
                ledger = Ledger.for_repo(repo)
                if u.path == "/api/samples":
                    return self.send(200, samples(root, q["ref"]))
                if u.path == "/api/report":
                    return self.send(200, cards.report(repo, ledger, q["ref"]))
                if u.path == "/api/explain":
                    return self.send(200, cards.explain(repo, ledger, q["ref"]))
                if u.path == "/api/card":
                    return self.send(200, cards.card(repo, ledger, q["id"]))
                if u.path.startswith("/api/jobs/"):
                    return self.send(200, job_view(root, u.path.rsplit("/", 1)[1], int(q.get("since", 0))))
                return self.send(404, {"error": "not found"})
            except KeyError as e:
                return self.send(404, {"error": f"unknown {e}"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/api/jobs/") and self.path.endswith(("/approve", "/cancel")):
                from ..store.queue import open_queue
                _, _, jid, action = self.path.rsplit("/", 3)
                try:
                    return self.send(200, getattr(open_queue(root), action)(jid).to_dict())
                except ValueError as e:
                    return self.send(400, {"error": str(e)})
            who = self.headers.get("X-Requested-By") or "web"
            if self.path == "/api/ingest":
                src, text = body.get("source", "").strip(), (body.get("text") or "").strip() or None
                if not src:
                    return self.send(400, {"error": "give an arXiv id or a post URL"})
                return self.send(200, enqueue(root, "ingest", {"source": src, **({"text": text} if text else {})}, who))
            if self.path == "/api/expand":
                models = [m.strip() for m in (body.get("models") or []) if m.strip()]
                ref = body.get("ref", "")
                if not ref or not models:
                    return self.send(400, {"error": "give a study and at least one model"})
                return self.send(200, enqueue(root, "expand", {"source": ref, "models": models}, who))
            if self.path == "/api/verify":
                ids = body.get("claims") or source_claims_to_verify(root, body.get("ref", ""))
                if not ids:
                    return self.send(400, {"error": "nothing to verify"})
                recipe = {"claims": ids, **({"force": True} if body.get("force") else {})}
                return self.send(200, {**enqueue(root, "verify", recipe, who), "claims": ids})
            return self.send(404, {"error": "not found"})

    return Handler


def serve(root: Path, port: int = 8765) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(root).resolve()))
    print(f"claimci: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
