"""A small local web app: drop a paper or post, see its claims, verify them.

Standard library only. Ingest calls the Claude API from this machine; verification runs
on the remote GPU box when .claimci-remote.json exists (claimci/remote.py), else locally.

    claimci serve --port 8765
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import cards
from .ledger import Ledger
from .loader import load_repo

PAGE = Path(__file__).with_name("web") / "index.html"
JOBS: dict[str, dict] = {}
VERIFY_LOCK = threading.Lock()


def overview(root: Path) -> dict:
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)
    sources: dict[str, dict] = {}
    for k, c in sorted(repo.claims.items()):
        if not c.is_source:
            continue
        src = c.provenance.get("source") or {}
        s = sources.setdefault(src.get("ref", k), {"ref": src.get("ref", k), "kind": src.get("kind"),
                                                   "title": src.get("title") or src.get("ref"), "claims": [],
                                                   "question": c.question})
        s["claims"].append({"id": k, "support": cards.card(repo, ledger, k)["support"]})
    return {"sources": list(sources.values()), "remote": (root / ".claimci-remote.json").exists()}


def source_claims_to_verify(root: Path, ref: str) -> list[str]:
    """Every test and measurement in the questions of this source's claims."""
    repo = load_repo(root)
    qs = {c.question for c in repo.claims.values() if c.is_source
          and (c.provenance.get("source") or {}).get("ref") == ref}
    return sorted(k for k, c in repo.claims.items() if not c.is_source and c.question in qs)


def start_job(kind: str, fn) -> str:
    jid = uuid.uuid4().hex[:8]
    job = JOBS[jid] = {"id": jid, "kind": kind, "lines": [], "done": False, "result": None, "error": None}

    def log(line):
        job["lines"].append(str(line))

    def run():
        try:
            job["result"] = fn(log)
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"
            log(traceback.format_exc())
        finally:
            job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jid


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
                if u.path == "/api/overview":
                    return self.send(200, overview(root))
                repo = load_repo(root)
                ledger = Ledger.for_repo(repo)
                if u.path == "/api/explain":
                    return self.send(200, cards.explain(repo, ledger, q["ref"]))
                if u.path == "/api/card":
                    return self.send(200, cards.card(repo, ledger, q["id"]))
                if u.path.startswith("/api/jobs/"):
                    job = JOBS[u.path.rsplit("/", 1)[1]]
                    since = int(q.get("since", 0))
                    return self.send(200, {**job, "lines": job["lines"][since:], "n": len(job["lines"])})
                return self.send(404, {"error": "not found"})
            except KeyError as e:
                return self.send(404, {"error": f"unknown {e}"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/ingest":
                from .pipeline import ingest_source
                src, text = body.get("source", "").strip(), (body.get("text") or "").strip() or None
                if not src:
                    return self.send(400, {"error": "give an arXiv id or a post URL"})
                jid = start_job("ingest", lambda log: ingest_source(root, src, text, log=log))
                return self.send(200, {"job": jid})
            if self.path == "/api/verify":
                ids = body.get("claims") or source_claims_to_verify(root, body.get("ref", ""))
                if not ids:
                    return self.send(400, {"error": "nothing to verify"})

                def run(log):
                    with VERIFY_LOCK:  # one model run at a time
                        if (root / ".claimci-remote.json").exists():
                            from .remote import verify as remote_verify
                            return remote_verify(root, ids, log, force=bool(body.get("force")))
                        from .backends import CpuBackend
                        from .cli import verify
                        verify(load_repo(root), CpuBackend(), claims=ids, log=log, force=bool(body.get("force")))
                jid = start_job("verify", run)
                return self.send(200, {"job": jid, "claims": ids})
            return self.send(404, {"error": "not found"})

    return Handler


def serve(root: Path, port: int = 8765) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(root).resolve()))
    print(f"claimci: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
