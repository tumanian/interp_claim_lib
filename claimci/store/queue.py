"""The job queue between the web app and the compute (DESIGN 23).

A job is a recipe (what to run: ingest a source, verify claims, expand a source to models) and its
state. The web app only enqueues; `claimci worker` claims and runs. Two backends behind one
interface: Postgres (CLAIMCI_DB_URL, e.g. Supabase; the shared, cross-machine one) and SQLite
(ledger/queue.sqlite; local-only and tests).

    pending -> approved -> running -> done | failed
                 \\-> cancelled (from pending or approved)

The same recipe is one job while it is pending, approved or running (dedupe on recipe_hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("ingest", "verify", "expand")
OPEN = ("pending", "approved", "running")
COLUMNS = ("id", "kind", "recipe", "recipe_hash", "status", "estimate", "cost", "requested_by",
           "created", "started", "finished", "error", "log", "worker")
JSON_COLUMNS = ("recipe", "estimate", "cost")


@dataclass
class Job:
    id: str
    kind: str
    recipe: dict
    recipe_hash: str
    status: str
    estimate: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    requested_by: str | None = None
    created: str | None = None
    started: str | None = None
    finished: str | None = None
    error: str | None = None
    log: str | None = None
    worker: str | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in COLUMNS}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recipe_hash(kind: str, recipe: dict) -> str:
    return hashlib.sha256(json.dumps({"kind": kind, **recipe}, sort_keys=True).encode()).hexdigest()[:16]


def _row(values) -> Job:
    d = dict(zip(COLUMNS, values))
    for k in JSON_COLUMNS:
        d[k] = json.loads(d[k]) if isinstance(d[k], str) else (d[k] or {})
    return Job(**d)


class Queue:
    """Shared logic; subclasses supply `_execute(sql, params) -> rows` with %s placeholders, and
    `_claim`. Every method is one statement or one transaction."""

    def add(self, kind: str, recipe: dict, requested_by: str | None = None, estimate: dict | None = None,
            approved: bool = False) -> Job:
        """Enqueue, or return the open job with the same recipe."""
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        h = recipe_hash(kind, recipe)
        open_ = self._execute(f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE recipe_hash = %s AND status IN %s", (h, OPEN))
        if open_:
            return _row(open_[0])
        job = Job(uuid.uuid4().hex[:12], kind, recipe, h, "approved" if approved else "pending",
                  estimate or {}, {}, requested_by, now())
        self._execute(f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))})",
                      tuple(json.dumps(getattr(job, k)) if k in JSON_COLUMNS else getattr(job, k) for k in COLUMNS))
        return job

    def get(self, job_id: str) -> Job | None:
        rows = self._execute(f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE id = %s", (job_id,))
        return _row(rows[0]) if rows else None

    def list(self, status: str | None = None, limit: int = 100) -> list[Job]:
        where, params = ("WHERE status = %s", (status,)) if status else ("", ())
        rows = self._execute(f"SELECT {', '.join(COLUMNS)} FROM jobs {where} ORDER BY created DESC LIMIT %s", params + (limit,))
        return [_row(r) for r in rows]

    def approve(self, job_id: str) -> Job:
        return self._move(job_id, "approved", from_=("pending",))

    def cancel(self, job_id: str) -> Job:
        return self._move(job_id, "cancelled", from_=("pending", "approved"), finished=now())

    def finish(self, job_id: str, status: str, cost: dict | None = None, error: str | None = None,
               log: str | None = None) -> Job:
        if status not in ("done", "failed"):
            raise ValueError(status)
        return self._move(job_id, status, from_=("running",), finished=now(), cost=json.dumps(cost or {}),
                          error=error, log=log)

    def requeue_stale(self, older_than_seconds: int) -> list[Job]:
        """Running jobs whose worker stopped heartbeating: back to approved, once (then failed)."""
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_seconds
        out = []
        for j in self.list("running"):
            hb = self._execute("SELECT seen FROM workers WHERE id = %s", (j.worker,))
            seen = datetime.strptime(hb[0][0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if hb else 0
            if seen < cutoff:
                if j.error == "requeued once":
                    out.append(self._move(j.id, "failed", from_=("running",), finished=now(), error="worker died twice"))
                else:
                    out.append(self._move(j.id, "approved", from_=("running",), started=None, worker=None, error="requeued once"))
        return out

    def heartbeat(self, worker_id: str, job_id: str | None = None) -> None:
        self._execute("DELETE FROM workers WHERE id = %s", (worker_id,))
        self._execute("INSERT INTO workers (id, seen, job) VALUES (%s, %s, %s)", (worker_id, now(), job_id))

    def workers(self) -> list[dict]:
        return [{"id": r[0], "seen": r[1], "job": r[2]} for r in self._execute("SELECT id, seen, job FROM workers ORDER BY seen DESC", ())]

    def claim(self, worker_id: str) -> Job | None:
        """The oldest approved job, now running under this worker; atomic across workers."""
        rows = self._claim(worker_id)
        return _row(rows[0]) if rows else None

    def _move(self, job_id: str, status: str, from_: tuple, **sets) -> Job:
        cols = ", ".join(["status = %s"] + [f"{k} = %s" for k in sets])
        rows = self._execute(f"UPDATE jobs SET {cols} WHERE id = %s AND status IN %s RETURNING {', '.join(COLUMNS)}",
                             (status, *sets.values(), job_id, from_))
        if not rows:
            cur = self.get(job_id)
            raise ValueError(f"job {job_id}: {'not found' if cur is None else f'is {cur.status}, cannot become {status}'}")
        return _row(rows[0])

    def _execute(self, sql: str, params: tuple) -> list:
        raise NotImplementedError

    def _claim(self, worker_id: str) -> list:
        raise NotImplementedError


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, recipe TEXT NOT NULL, recipe_hash TEXT NOT NULL,
        status TEXT NOT NULL, estimate TEXT, cost TEXT, requested_by TEXT, created TEXT NOT NULL,
        started TEXT, finished TEXT, error TEXT, log TEXT, worker TEXT)""",
    "CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created)",
    "CREATE INDEX IF NOT EXISTS jobs_recipe_hash ON jobs (recipe_hash)",
    "CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, seen TEXT NOT NULL, job TEXT)",
]


class SqliteQueue(Queue):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        for stmt in SCHEMA:
            self.conn.execute(stmt)

    def _execute(self, sql: str, params: tuple) -> list:
        sql, params = _expand_in(sql, params)
        return self.conn.execute(sql.replace("%s", "?"), params).fetchall()

    def _claim(self, worker_id: str) -> list:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT id FROM jobs WHERE status = 'approved' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return []
            return self.conn.execute(f"UPDATE jobs SET status = 'running', started = ?, worker = ? WHERE id = ? RETURNING {', '.join(COLUMNS)}",
                                     (now(), worker_id, row[0])).fetchall()
        finally:
            self.conn.execute("COMMIT")


class PostgresQueue(Queue):
    """Tables live in the `claimci` schema of the database at `url`."""

    def __init__(self, url: str):
        import psycopg
        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS claimci")
        self.conn.execute("SET search_path TO claimci")
        for stmt in SCHEMA:
            self.conn.execute(stmt)

    def _execute(self, sql: str, params: tuple) -> list:
        sql, params = _expand_in(sql, params)
        cur = self.conn.execute(sql, params)
        return cur.fetchall() if cur.description else []

    def _claim(self, worker_id: str) -> list:
        return self._execute(f"""UPDATE jobs SET status = 'running', started = %s, worker = %s
            WHERE id = (SELECT id FROM jobs WHERE status = 'approved' ORDER BY created LIMIT 1 FOR UPDATE SKIP LOCKED)
            RETURNING {', '.join(COLUMNS)}""", (now(), worker_id))


def _expand_in(sql: str, params: tuple) -> tuple[str, tuple]:
    """`IN %s` with a tuple parameter becomes `IN (%s, %s, ...)` on both backends."""
    out, flat = [], []
    parts = sql.split("%s")
    for i, p in enumerate(params):
        if isinstance(p, tuple):
            out.append(parts[i].rstrip() + " (" + ", ".join(["%s"] * len(p)) + ")" if parts[i].rstrip().endswith("IN") else parts[i] + "(" + ", ".join(["%s"] * len(p)) + ")")
            flat += list(p)
        else:
            out.append(parts[i] + "%s")
            flat.append(p)
    return "".join(out) + parts[len(params)], tuple(flat)


def open_queue(root: Path | str = ".") -> Queue:
    """Postgres when CLAIMCI_DB_URL is set (loaded from <root>/.env by the CLI), else SQLite in ledger/."""
    url = os.environ.get("CLAIMCI_DB_URL")
    return PostgresQueue(url) if url else SqliteQueue(Path(root) / "ledger" / "queue.sqlite")
