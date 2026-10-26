"""Runner: execute a Plan on a Backend and record results in the ledger."""
from __future__ import annotations

import hashlib
import importlib
import signal
import socket
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from . import __version__
from .ledger import Ledger
from .planner import Group, Plan, is_intervention
from .spec import Repo, Result, evaluate

CANARY_N = 4
CANARY_ATOL = 1e-4


class ClaimTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float | None):
    """Per-claim wall clock, via SIGALRM where available (main thread on Unix)."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def fire(signum, frame):
        raise ClaimTimeout(f"claim exceeded {seconds}s")

    try:
        old = signal.signal(signal.SIGALRM, fire)
    except ValueError:  # not the main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def resolve_entrypoint(ep: str):
    mod, _, fn = ep.partition(":")
    return getattr(importlib.import_module(mod), fn)


class Runner:
    def __init__(self, repo: Repo, ledger: Ledger, backend, claim_timeout: float | None = 600,
                 mem_budget: int | None = None, log=print):
        self.repo, self.ledger, self.backend = repo, ledger, backend
        self.claim_timeout, self.mem_budget, self.log = claim_timeout, mem_budget, log
        self.run_id = uuid.uuid4().hex[:8]
        self.outputs: dict[tuple[str, str], Result] = {}

    def execute(self, plan: Plan) -> list[Result]:
        results: list[Result] = []
        by_model_rev: dict[tuple, list[Group]] = {}
        for g in plan.groups:
            by_model_rev.setdefault((g.model, g.revision), []).append(g)
        for (model, rev), groups in by_model_rev.items():
            t0 = time.perf_counter()
            handle = self.backend.load(model, rev)
            load_share = (time.perf_counter() - t0) / sum(len(g.claims) for g in groups)
            try:
                for g in groups:
                    out = self.run_group(handle, g, plan.mode, load_share)
                    results += out
                    if any(r.status == "hook_leak" for r in out):  # never reuse a leaked model
                        self.backend.unload(handle)
                        handle = self.backend.load(model, rev)
            finally:
                self.backend.unload(handle)
        return results

    # -- one group = one shared forward pass (shared mode) --------------------
    def run_group(self, handle, g: Group, mode: str, load_share: float) -> list[Result]:
        from .adapters import HFAdapter
        from .backends import Ctx, acts_bytes
        from .datasets import load_dataset

        ds = load_dataset(g.dataset)
        cfg = handle.model.config
        need = acts_bytes(cfg.num_hidden_layers, len(ds.tokens), len(ds.tokens[0]),
                          cfg.hidden_size, cfg.num_attention_heads)
        if self.mem_budget and need > self.mem_budget:
            raise MemoryError(f"shared cache needs {need} bytes > --mem-budget {self.mem_budget}; "
                              "sharding is phase 2")
        adapter = HFAdapter(handle)
        shared, shared_cost = None, (0.0, 0.0)
        runnable: set[str] = set()  # claims whose dependencies hold or will run in this group
        for cid in g.claims:  # topological order
            if all(d in runnable or self.dependency_holds(d, g.revision)
                   for d in self.repo.claims[cid].depends_on):
                runnable.add(cid)
        if mode == "shared" and runnable:
            t0 = time.perf_counter()
            shared = self.backend.run(handle, ds)
            shared_cost = (1.0 / len(g.claims), (time.perf_counter() - t0) / len(g.claims))
        out, aborted = [], None
        for cid in g.claims:
            if aborted:
                r = self.result(cid, g, "hook_leak", reason=aborted)
            else:
                r = self.run_claim(cid, g, handle, ds, adapter, shared, Ctx)
                r.cost["forward_passes"] += shared_cost[0]
                r.cost["seconds"] += shared_cost[1] + load_share
                if r.status == "hook_leak":
                    aborted = f"group aborted: hook leak after {cid}"
            r.cost["seconds"] = round(r.cost["seconds"], 6)
            r.cost["mode"] = mode
            self.ledger.put(r)
            self.outputs[(cid, g.revision)] = r
            self.log(self.describe(r))
            out.append(r)
        return out

    def dependency_holds(self, d: str, rev: str) -> bool:
        r = self.outputs.get((d, rev)) or self.ledger.get(d, self.repo.claim_key(d, rev))
        return r is not None and r.status == "ok" and bool(evaluate(self.repo.claims[d].expect, r.value))

    def dependency_block(self, cid: str, rev: str):
        """(None, bindings_by_parent) if runnable, else (reason, None)."""
        found = {}
        for d in self.repo.claims[cid].depends_on:
            r = self.outputs.get((d, rev)) or self.ledger.get(d, self.repo.claim_key(d, rev))
            if r is None:
                return f"dependency {d} has no result", None
            if r.status != "ok":
                return f"dependency {d} is {r.status}", None
            if not evaluate(self.repo.claims[d].expect, r.value):
                return f"dependency {d} does not hold (value={r.value})", None
            found[d] = r.bindings
        return None, found

    def run_claim(self, cid, g: Group, handle, ds, adapter, shared, Ctx) -> Result:
        c = self.repo.claims[cid]
        reason, parents = self.dependency_block(cid, g.revision)
        if reason:
            return self.result(cid, g, "blocked", reason=reason)
        inputs = {k: parents[p][e] for k, (p, e) in c.input_refs().items()}
        intervention = is_intervention(self.repo, cid)
        ctx = Ctx(self.backend, handle, ds, adapter)
        passes, canary, leak, t0 = 0.0, 0.0, None, time.perf_counter()
        try:
            with time_limit(self.claim_timeout):
                acts = shared
                if acts is None:  # naive mode: no sharing
                    acts, passes = self.backend.run(handle, ds), 1.0
                fn = resolve_entrypoint(c.code["entrypoint"])
                if intervention:
                    value, bindings = fn(acts, ctx, **inputs, **c.params), dict(inputs)
                else:
                    value, bindings = fn(acts, **inputs, **c.params)
                missing = set(c.exports) - set(bindings)
                if missing:
                    raise ValueError(f"experiment did not return exports {sorted(missing)}")
                value = round(float(value), 6)
                if intervention:
                    canary, leak = 1.0, self.canary(handle, ds, acts)
        except Exception as e:
            tb = traceback.format_exc()
            r = self.result(cid, g, "error", reason=f"{type(e).__name__}: {e}",
                            traceback_hash=hashlib.sha256(tb.encode()).hexdigest()[:16])
        else:
            r = self.result(cid, g, "ok", value=value, holds=evaluate(c.expect, value), bindings=bindings)
            if c.status == "proposed":  # a promotion run: compare to the source's stated value
                r.reproduction = self.reproduction(cid, value)
            if leak:
                r.status, r.reason, r.holds = "hook_leak", leak, None
        r.cost.update(forward_passes=passes + ctx.passes, canary_passes=canary,
                      seconds=time.perf_counter() - t0)
        return r

    def canary(self, handle, ds, acts) -> str | None:
        import torch
        probe = self.backend.run(handle, ds.head(CANARY_N))
        if torch.allclose(probe.loss, acts.loss[:CANARY_N], atol=CANARY_ATOL):
            return None
        diff = (probe.loss - acts.loss[:CANARY_N]).abs().max().item()
        return f"canary loss on first {CANARY_N} sequences differs from shared cache by {diff:.6f}"

    def reproduction(self, cid: str, value: float) -> dict | None:
        c = self.repo.claims[cid]
        target = (c.provenance.get("source") or {}).get("stated_value")
        if target is None or c.tolerance is None:
            return None
        return {"target": target, "within_tolerance": abs(value - target) <= c.tolerance}

    def result(self, cid: str, g: Group, status: str, **kw) -> Result:
        c = self.repo.claims[cid]
        return Result(
            claim=cid, claim_key=self.repo.claim_key(cid, g.revision), status=status,
            model=g.model, revision=g.revision, code_version=c.code_version,
            data_hash=g.dataset.hash, dataset=g.dataset.short, lineage=self.repo.lineage(cid),
            cost={"forward_passes": 0.0, "canary_passes": 0.0, "seconds": 0.0,
                  "backend": getattr(self.backend, "name", "?")},
            run={"runner": f"claimci {__version__}", "host": socket.gethostname(), "run_id": self.run_id,
                 "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, **kw)

    @staticmethod
    def describe(r: Result) -> str:
        detail = f"value={r.value} holds={r.holds} bindings={r.bindings}" if r.status == "ok" else r.reason
        return f"{r.revision:<10} {r.claim:<32} {r.status:<9} {detail}"
