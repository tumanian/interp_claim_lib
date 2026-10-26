"""Planner: claims x revisions -> groups sharing one forward pass. Never loads a model."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ledger import Ledger
from .spec import DataVersion, Repo


def is_intervention(repo: Repo, cid: str) -> bool:
    return repo.evidence[repo.claims[cid].evidence].causal


def topo_order(repo: Repo, ids) -> list[str]:
    """ids sorted so every claim comes after its depends_on (ties broken by id)."""
    ids, order, done = set(ids), [], set()

    def visit(c: str) -> None:
        if c in done:
            return
        done.add(c)
        for d in sorted(repo.claims[c].depends_on):
            if d in repo.claims:
                visit(d)
        if c in ids:
            order.append(c)

    for c in sorted(ids):
        visit(c)
    return order


@dataclass
class Group:
    model: str
    revision: str
    dataset: DataVersion
    claims: list[str]  # topologically sorted
    interventional: list[str] = field(default_factory=list)

    @property
    def pass_key(self) -> tuple:
        return (self.model, self.revision, self.dataset.ident)

    def passes(self, mode: str) -> int:
        clean = 1 if mode == "shared" else len(self.claims)
        return clean + len(self.interventional)


@dataclass
class Plan:
    groups: list[Group]
    skipped: list[tuple[str, str]]  # (claim_id, revision) ledger hits
    mode: str = "shared"
    sec_per_pass: float | None = None

    @property
    def to_run(self) -> int:
        return sum(len(g.claims) for g in self.groups)

    @property
    def forward_passes(self) -> int:
        return sum(g.passes(self.mode) for g in self.groups)

    def render(self) -> str:
        lines = []
        for g in self.groups:
            n_int = len(g.interventional)
            clean = "1 shared" if self.mode == "shared" else f"{len(g.claims)} naive"
            lines.append(f"checkpoint {g.revision:<10} {g.dataset.short:<28} {len(g.claims)} claims  "
                         f"{clean} + {n_int} intervention pass{'es' * (n_int != 1)}")
        est = "n/a (no cost history for this model)"
        if self.sec_per_pass is not None:
            m, s = divmod(round(self.sec_per_pass * self.forward_passes), 60)
            est = f"{m}m {s:02d}s"
        lines.append(f"skipped (ledger hit): {len(self.skipped)}   to run: {self.to_run}   "
                     f"forward passes: {self.forward_passes}   est: {est}")
        return "\n".join(lines)


def _sec_per_pass(ledger: Ledger, models: set[str]) -> float | None:
    secs = passes = 0.0
    for r in ledger.rows:
        if r.model in models and r.cost.get("forward_passes"):
            secs += r.cost.get("seconds", 0.0)
            passes += r.cost["forward_passes"]
    return secs / passes if passes else None


def make_plan(repo: Repo, ledger: Ledger, revisions: list[str] | None = None,
              claim_ids: list[str] | None = None, mode: str = "shared", force: bool = False) -> Plan:
    if claim_ids:
        unknown = set(claim_ids) - set(repo.claims)
        if unknown:
            raise KeyError(f"unknown claims: {sorted(unknown)}")
        selected = set(claim_ids).union(*(repo.ancestors(c) for c in claim_ids))
    else:
        selected = set(repo.claims)
    rerunnable = {c for c in selected if repo.evidence[repo.claims[c].evidence].rerunnable}
    order = topo_order(repo, rerunnable)
    revs = revisions or sorted({repo.claims[c].revision for c in order})
    force = force or mode == "naive"  # naive is a baseline measurement: always executes
    groups: list[Group] = []
    skipped: list[tuple[str, str]] = []
    for rev in revs:  # all datasets for one checkpoint before the next
        by_key: dict[tuple, Group] = {}
        for cid in order:
            c = repo.claims[cid]
            if not force and ledger.get(cid, repo.claim_key(cid, rev)):
                skipped.append((cid, rev))
                continue
            g = by_key.setdefault((c.model, c.dataset.ident), Group(c.model, rev, c.dataset, []))
            g.claims.append(cid)
            if is_intervention(repo, cid):
                g.interventional.append(cid)
        groups += by_key.values()  # insertion order follows the topological order
    sec = _sec_per_pass(ledger, {g.model for g in groups})
    return Plan(groups, skipped, mode, sec)
