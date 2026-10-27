"""Append-only JSONL ledger of results, indexed in memory on (claim_id, claim_key)."""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from ..schema.claim import Result
from ..schema.repo import Repo

RESULT_FIELDS = {f.name for f in fields(Result)}


class Ledger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.rows: list[Result] = []
        self.index: dict[tuple[str, str], list[Result]] = {}
        if self.path.exists():
            for n, line in enumerate(self.path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    self._add(Result(**{k: v for k, v in json.loads(line).items() if k in RESULT_FIELDS}))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{self.path}:{n}: malformed ledger row: {e}") from e

    @classmethod
    def for_repo(cls, repo: Repo) -> "Ledger":
        return cls(repo.root / "ledger" / "results.jsonl")

    def _add(self, r: Result) -> None:
        self.rows.append(r)
        self.index.setdefault((r.claim, r.claim_key), []).append(r)

    def put(self, r: Result) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
        self._add(r)

    def get(self, claim_id: str, claim_key: str, ok_only: bool = True) -> Result | None:
        """Latest row for this key; by default only rows with status ok count as a result."""
        rows = [r for r in self.index.get((claim_id, claim_key), []) if r.status == "ok" or not ok_only]
        return rows[-1] if rows else None

    def history(self, claim_id: str, revision: str | None = None) -> list[Result]:
        return [r for r in self.rows if r.claim == claim_id and (revision is None or r.revision == revision)]

    def stale(self, repo: Repo, revision: str | None = None, claim_ids=None) -> list[str]:
        """Claims whose current key has no ok row. Never stored; always computed."""
        ids = sorted(claim_ids if claim_ids is not None else (k for k in repo.claims if not repo.claims[k].is_source))
        return [c for c in ids if self.get(c, repo.claim_key(c, revision)) is None]

    def stale_reason(self, repo: Repo, cid: str, revision: str | None = None) -> str:
        """Why a stale claim is stale, compared to its latest ok row on the same revision."""
        rev = revision or repo.claims[cid].revision
        prior = [r for r in self.history(cid, rev) if r.status == "ok"]
        if not prior:
            return "never run"
        last, c = prior[-1], repo.claims[cid]
        old, new = last.lineage, repo.lineage(cid)
        if set(old) != set(new) or any(old[k][0] != new[k][0] for k in new):
            return "code changed"
        if any(old[k][1] != new[k][1] for k in new):
            return "data changed"
        if last.model != c.model:
            return "model changed"
        return "parameters or inputs changed"


def current_row(repo: Repo, ledger: Ledger, cid: str, rev: str) -> Result | None:
    """Latest row (any status) for the claim's current key at this revision."""
    rows = ledger.index.get((cid, repo.claim_key(cid, rev)), [])
    return rows[-1] if rows else None


def gated_by(repo: Repo, ledger: Ledger, cid: str, rev: str) -> list[str]:
    """Ancestors that do not hold at this revision under their current expect.

    Values are always recorded; whether a dependent counts is decided here, on read,
    so editing a parent's threshold re-gates without rerunning."""
    out = []
    for a in sorted(repo.ancestors(cid)):
        r = current_row(repo, ledger, a, rev)
        if r is None or r.status != "ok" or repo.claims[a].holds(r.value) is False:
            out.append(a)
    return out
