"""Runner: execute a Plan on a Backend and record results in the ledger.

The loop: for each model (loaded once), for each group (one shared pass over its data), for each
claim in dependency order: wait for its parents, call its experiment under the guards, record a
row. How experiments are called is in calling.py, the guards in guards.py, rows in rows.py.
"""
from __future__ import annotations

import time
import traceback
import uuid

from ..schema.claim import Result
from ..schema.repo import Repo
from ..store.ledger import Ledger
from .calling import call, resolve_entrypoint
from .guards import contamination, time_limit
from .planner import Group, Plan, is_intervention
from .rows import describe, row, run_info


class Runner:
    def __init__(self, repo: Repo, ledger: Ledger, backend, claim_timeout: float | None = 600,
                 mem_budget: int | None = None, log=print):
        self.repo, self.ledger, self.backend = repo, ledger, backend
        self.claim_timeout, self.mem_budget, self.log = claim_timeout, mem_budget, log
        self.run_id = uuid.uuid4().hex[:8]
        from ..store.artifacts import Store
        self.artifacts = Store(ledger.path.parent / "artifacts")
        self.outputs: dict[tuple[str, str], Result] = {}

    def execute(self, plan: Plan) -> list[Result]:
        results: list[Result] = []
        by_model: dict[tuple, list[Group]] = {}
        for g in plan.groups:
            by_model.setdefault((g.model, g.revision, tuple(sorted(g.opts.items()))), []).append(g)
        for (model, rev, opts), groups in by_model.items():
            load = lambda: self.backend.load(model, rev, dict(opts)) if opts else self.backend.load(model, rev)
            t0 = time.perf_counter()
            handle = load()
            load_share = (time.perf_counter() - t0) / sum(len(g.claims) for g in groups)
            try:
                for g in groups:
                    out = self.run_group(handle, g, plan.mode, load_share)
                    results += out
                    if any(r.status == "contaminated" for r in out):  # never reuse a contaminated model
                        self.backend.unload(handle)
                        handle = load()
            finally:
                self.backend.unload(handle)
        return results

    # -- one group = one shared forward pass (shared mode) --------------------
    def run_group(self, handle, g: Group, mode: str, load_share: float) -> list[Result]:
        from .adapters import HFAdapter
        from .backends import acts_bytes
        from ..store.datasets import load_dataset

        ds = load_dataset(g.dataset)
        if not ds.is_prompts and self.mem_budget:
            cfg = handle.model.config
            need = acts_bytes(cfg.num_hidden_layers, len(ds.tokens), len(ds.tokens[0]), cfg.hidden_size,
                              cfg.num_attention_heads)
            if need > self.mem_budget:
                raise MemoryError(f"shared cache needs {need} bytes > --mem-budget {self.mem_budget}; sharding is phase 2")
        adapter = HFAdapter(handle)
        runnable: set[str] = set()  # claims whose parents have a value or will run in this group
        for cid in g.claims:  # topological order
            if all(d in runnable or self.parent_ok(d, g.revision) for d in self.repo.claims[cid].depends_on):
                runnable.add(cid)
        shared, shared_cost = None, (0.0, 0.0)
        if mode == "shared" and runnable:
            t0 = time.perf_counter()
            shared = self.backend.run(handle, ds)
            shared_cost = (1.0 / len(g.claims), (time.perf_counter() - t0) / len(g.claims))
        out, aborted = [], None
        for cid in g.claims:
            if aborted:
                r = self.row(cid, g, "contaminated", reason=aborted)
            else:
                r = self.run_claim(cid, g, handle, ds, adapter, shared)
                r.cost["forward_passes"] += shared_cost[0]
                r.cost["seconds"] += shared_cost[1] + load_share
                if r.status == "contaminated":
                    aborted = f"group aborted: hook leak after {cid}"
            r.cost.update(seconds=round(r.cost["seconds"], 6), mode=mode)
            self.ledger.put(r)
            self.outputs[(cid, g.revision)] = r
            self.log(describe(r, self.repo.claims[r.claim].holds(r.value, r.interval)))
            out.append(r)
        return out

    # -- one claim ---------------------------------------------------------------
    def parent(self, d: str, rev: str) -> Result | None:
        """A parent's result at this revision: from this run, else from the ledger."""
        return self.outputs.get((d, rev)) or self.ledger.get(d, self.repo.claim_key(d, rev))

    def parent_ok(self, d: str, rev: str) -> bool:
        r = self.parent(d, rev)
        return r is not None and r.status == "ok"

    def run_claim(self, cid, g: Group, handle, ds, adapter, shared) -> Result:
        from .backends import Ctx
        c = self.repo.claims[cid]
        for d in c.depends_on:   # a parent that does not hold still binds: gating is derived on read
            r = self.parent(d, g.revision)
            if r is None or r.status != "ok":
                return self.row(cid, g, "blocked", reason=f"dependency {d} {'has no result' if r is None else 'is ' + r.status}")
        inputs = {k: self.parent(p, g.revision).bindings[e] for k, (p, e) in c.input_refs().items()}
        intervention = is_intervention(self.repo, cid)
        ctx = Ctx(self.backend, handle, ds, adapter, artifacts=self.artifacts)
        passes, canary, leak, t0 = 0.0, 0.0, None, time.perf_counter()
        try:
            with time_limit(self.claim_timeout):
                ctx.acts = shared
                if shared is None:  # naive mode: no sharing
                    ctx.acts, passes = self.backend.run(handle, ds), 1.0
                value, bindings, interval = call(resolve_entrypoint(c.entrypoint), ctx, c, inputs, intervention)
                if intervention:
                    canary, leak = 1.0, contamination(self.backend, handle, ds, ctx.acts)
        except Exception as e:
            r = self.row(cid, g, "error", reason=f"{type(e).__name__}: {e}",
                         traceback=self.artifacts.save_json({"traceback": traceback.format_exc()}))
        else:
            r = self.row(cid, g, "contaminated" if leak else "ok", value=value, bindings=bindings,
                         interval=list(interval) if interval else None, reason=leak)
        r.cost.update(forward_passes=passes + ctx.passes, canary_passes=canary, seconds=time.perf_counter() - t0)
        return r

    def row(self, cid: str, g: Group, status: str, **kw) -> Result:
        return row(self.repo, cid, g, status, run_info(self.run_id), getattr(self.backend, "name", "?"), **kw)
