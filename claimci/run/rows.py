"""Ledger rows as the runner writes them, and the one-line log of each."""
from __future__ import annotations

import socket
from datetime import datetime, timezone

from .. import __version__
from ..schema.claim import Result


def run_info(run_id: str) -> dict:
    return {"runner": f"claimci {__version__}", "host": socket.gethostname(), "run_id": run_id,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def row(repo, cid: str, group, status: str, run: dict, backend: str, **kw) -> Result:
    """A result: what ran (key, code, data, model) and what came out. Verdicts are not stored."""
    return Result(claim=cid, claim_key=repo.claim_key(cid, group.revision), status=status,
                  model=group.model, revision=group.revision, code_version=repo.claims[cid].code_version,
                  data_hash=group.dataset.hash, lineage=repo.lineage(cid),
                  cost={"forward_passes": 0.0, "canary_passes": 0.0, "seconds": 0.0, "backend": backend},
                  run=dict(run), **kw)


def describe(r: Result, holds: bool | None = None) -> str:
    detail = f"value={r.value} holds={holds} bindings={r.bindings}" if r.status == "ok" else r.reason
    return f"{r.revision:<10} {r.claim:<32} {r.status:<9} {detail}"
