# claimci runner spec (v0.2)

The runner turns a set of claim files into a ledger of results, running as few
forward passes as possible. CPU is the only backend in v0; the backend is an
interface so GPU and remote workers slot in later (phase 3).

## Pipeline

```
load ──> validate ──> plan ──> execute ──> record ──> report
```

### 1. load
- Read `claims/*.yaml`, `questions/*.yaml`, `evidence.yaml`.
- Resolve `data.inherit` chains to concrete data versions.
- Resolve `code.version`: `git:<sha>` for external repos, or compute
  `sha(source + params)` for local entrypoints. Record which.

### 2. validate (fail fast, before any model loads)
- Typing rules from SPEC.md (no TAS/NAS/IEA dependencies, no ICP under causal, ...).
- Dependency graph is a DAG. Every `inputs` reference names an `exports` of a
  `depends_on` claim.
- Every rerunnable claim has `metric`, `expect`, `code.entrypoint`.
- Datasets materialize deterministically: generate, hash, compare to the
  declared `hash`. Mismatch is a hard error, not a warning.

### 3. plan
Produce an execution plan, printable without running anything (`claimci plan`).

- **Claim key** = `sha(code.version + params + data.hash)`.
- **Skip** any claim whose key is already in the ledger with a result
  (unless `--force`).
- **Group** remaining claims by **pass key** = `(model, revision, dataset hash)`.
  One group = one shared forward pass.
- **Order** groups to minimize model loads: all datasets for one checkpoint
  before moving to the next. Model loads dominate CPU wall time.
- Within a group, **topologically sort** by `depends_on`. Observational claims
  (IOB, ICP) read the shared cache. Interventional claims (IMP, ISS, ICA)
  each get their own forward pass with hooks, after their dependencies have
  produced bindings.
- Estimate cost per group from the ledger's historical `cost` records for the
  same model; print total forward passes and estimated seconds.

Plan output (example):
```
checkpoint step1000   repeated_random@7d1e  3 claims  1 shared + 1 intervention pass
checkpoint step10000  repeated_random@7d1e  3 claims  1 shared + 1 intervention pass
...
skipped (ledger hit): 12   to run: 18   forward passes: 12   est: 4m 10s
```

### 4. execute
- **Backend interface** (`claimci.run.backends`):
  ```
  class Backend:
      def load(model, revision) -> ModelHandle
      def run(handle, dataset, hooks=()) -> Acts          # one forward pass
      def unload(handle)
  ```
  v0 ships `CpuBackend` (HF transformers, eager attention, `torch.no_grad`,
  thread count pinned). GPU, MPS, and remote backends are phase 3 and must
  satisfy the same interface. Backend choice is a CLI flag, never in a claim.
- **Adapters** (`claimci.run.adapters`) wrap a `ModelHandle` for external code:
  `hf` (raw), `transformer_lens` (HookedTransformer over the same weights,
  cache injected so `run_with_cache` is not called by the claim), `nnsight`
  (phase 2). Each claim's entrypoint declares which adapter it needs.
- **Shared cache**: `Acts` holds residual stream per layer, attention
  patterns, per-token loss, lengths, and dataset metadata. Built once per
  group, handed to every observational claim, discarded after the group.
  Memory budget check: if `L × N × T × D × 4 bytes` exceeds `--mem-budget`,
  split the dataset into shards and run claims shard-wise, aggregating by
  the claim's declared `aggregate` (mean by default).
- **Isolation**: every hook is registered inside a context manager and
  removed on exit, exception or not. After each interventional claim the
  runner runs a cheap canary (loss on the first 4 sequences) against the
  shared cache value; a mismatch aborts the group with `hook_leak`.
- **Determinism**: seeds fixed, `torch.use_deterministic_algorithms(True)`
  where supported, thread count pinned. Results are stored to 6 decimals.
- **Failure**: a claim that raises records `status: error` with the
  traceback hash, and does not block siblings. Dependents of a failed or
  non-holding claim are recorded as `status: blocked` with the reason.
- **Timeouts**: per-claim wall clock (`--claim-timeout`, default 10 min on CPU).

### 5. record
- Ledger is append-only JSONL (`ledger/results.jsonl`), one row per
  (claim_key, run). SQLite index built on load for queries (phase 2).
- Row fields: claim id, claim_key, code version, data hash, model, revision,
  dataset, value, holds, bindings, status, reproduction block if a promotion
  run, cost {forward_passes, seconds, backend}, runner version, timestamp.
- **Promotion**: if the claim is `proposed` with a `stated_value`, and the
  value is within `tolerance`, the runner writes `reproduced_from` and
  updates `evidence` and `status` in the claim file (the only field writes
  the runner ever makes to a claim file, and it does them as a separate
  commit).
- **Staleness** is never stored; it is computed: a claim is stale when its
  current key has no ledger row.

### 6. report
- `claimci matrix --question q.induction_formation`: claims × revisions,
  cell = value and pass/fail, rendered as markdown table and PNG.
- `claimci throughput`: claims verified, forward passes, seconds, claims per
  minute, and the same numbers for `--mode naive` (one pass per claim) from
  the same run, so the shared-pass saving is a measured ratio.
- `claimci status`: counts by status and evidence code; list of stale claims
  and why (code changed / data changed / never run).

## CLI

```
claimci validate                      # steps 1–2, no model loads
claimci plan [--revisions ...]        # step 3, prints plan
claimci verify [--revisions ...] [--claims ...] [--mode shared|naive] [--force]
claimci matrix --question <id>
claimci throughput
claimci status
```

## Invalidation semantics (what reruns)

| Change | Reruns |
|---|---|
| Edit a claim's experiment code or params | that claim, and dependents |
| Edit a dataset generator or its params | every claim on that dataset |
| Add a checkpoint | every claim, that checkpoint only |
| Edit `expect` | nothing; `holds` is re-evaluated from stored value |
| Edit `statement` or `provenance` | nothing |
| Runner version bump | nothing by default; `--force` to re-baseline |

## CPU constraints for v0

- Models: Pythia-70M to 410M, GPT-2 small. 2 cores / 7 GB here; 410M with
  the repeated-token set (48 × 40 tokens) fits in memory; attention patterns
  for 24 layers × 16 heads × 48 × 40 × 40 floats ≈ 120 MB.
- Checkpoints downloaded on demand, kept in the HF cache; `--offline` fails
  if any are missing rather than downloading.
- Target for the toy: 3 claims × 8 checkpoints of Pythia-160M under 10
  minutes wall clock including downloads.

## Phase plan

- **v0 (now):** CPU backend, HF adapter, local experiments, JSONL ledger,
  matrix + throughput reports, induction question end to end.
- **Phase 2:** TransformerLens adapter with cache injection, SQLite index,
  sharded cache, external repo entrypoints with lockfiles, ingestion agent
  writing TAS claims.
- **Phase 3:** GPU/MPS backend, remote workers (one group = one job),
  on-disk activation cache keyed by pass key, GPU-hour accounting in `cost`.
