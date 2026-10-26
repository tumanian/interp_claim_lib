# Build claimci v0

You are building `claimci`, a small CI system for mechanistic interpretability claims. A claim is experiment code applied to versioned data; verification is a rerun. Read `SPEC.md` (claim format, evidence codes, lifecycle) and `RUNNER.md` (pipeline, planner, backend interface, invalidation rules) before writing any code. They are the source of truth; where this prompt and the specs disagree, follow the specs and tell me.

Target for v0: one research question ("Does this model form induction heads, and when in training?"), three chained claims, run across Pythia training checkpoints on CPU, with a ledger, a validity matrix, and a throughput report.

Work in steps. **Do not start a step until the previous step's verification passes.** After each step, print the verification output and stop for my review.

Environment: Python 3.11+, CPU only, `torch` (CPU wheel), `transformers`, `pyyaml`, `numpy`, `pytest`. Use `EleutherAI/pythia-70m` for all development and tests; it loads in under 10 s. Switch to `pythia-160m` only in step 8. Pin `torch.set_num_threads` to the machine's core count. Never download a checkpoint inside a test; tests use the smallest revision already in the HF cache or skip with a clear message.

Repo layout:

```
claimci/
  claimci/
    __init__.py
    spec.py          # dataclasses: Claim, Question, EvidenceCode, DataVersion, Result
    evidence.yaml    # the evidence ontology
    datasets.py      # deterministic generators + content hash
    backends.py      # Backend interface, CpuBackend, Acts, hook context manager
    adapters.py      # hf adapter only in v0
    ledger.py
    planner.py
    runner.py
    report.py
    cli.py
    experiments/
      induction.py
  claims/*.yaml
  questions/*.yaml
  ledger/.gitkeep
  tests/
  pyproject.toml
  README.md
  SPEC.md
  RUNNER.md
```

---

## Step 1: spec types and loading

Implement `spec.py`, `evidence.yaml`, and loaders for `claims/*.yaml` and `questions/*.yaml`. Write the three induction claim files and one question file exactly as the examples in SPEC.md, with `provenance.source.ref: arXiv:2209.11895`.

Claims:
- `mech.prev_token_head` (IOB, exports `layer, head`)
- `mech.induction_head` (IOB, depends on prev_token_head, exports `layer, head`)
- `causal.induction_head_ablation` (IMP, depends on induction_head, inputs `layer, head` from it, `data.inherit: mech.induction_head`)

**Verify:**
- `pytest tests/test_spec.py` passes, covering: all three claims load; `data.inherit` resolves to the parent's concrete data version; `claim_key` is identical across two loads; changing `expect` in a claim file does not change `claim_key`; changing `code.params` does.
- `claimci validate` exits 0 on the repo as written.
- Introduce each of these errors in a temp copy and confirm `claimci validate` exits non-zero with a message naming the claim: a cycle in `depends_on`; an `inputs` reference to a name the parent does not export; a `TAS` claim listed in another claim's `depends_on`; an `ICP` claim under an `IMP` claim's `depends_on`.

## Step 2: datasets

Implement `datasets.py` with one generator, `repeated_random_tokens(n, half, seed, vocab_lo, vocab_hi)`, returning token id sequences of length `2*half` where the second half repeats the first. Content hash = sha256 of the canonical JSON of the materialized data.

**Verify:**
- Same params produce the same hash across two processes (run twice via subprocess).
- Changing `seed` changes the hash.
- The hash declared in the claim files matches the generated hash; `claimci validate` fails if you edit the declared hash by one character.

## Step 3: backend and activation cache

Implement `backends.py`: the `Backend` interface from RUNNER.md, `CpuBackend` using HF transformers with `attn_implementation="eager"`, `output_hidden_states=True`, `output_attentions=True`, `torch.no_grad()`. `Acts` holds hidden states `[L+1, N, T, D]`, attention `[L, N, H, T, T]`, per-token loss `[N, T-1]`, lengths, and dataset metadata. Hooks are passed as `(module, fn)` pairs and registered inside a context manager that removes them on exit, including on exception.

**Verify:**
- Two consecutive `run` calls on the same model and dataset produce bitwise-identical `Acts` (compare with `torch.equal`).
- A hook that raises mid-forward leaves no handles registered: assert `len(module._forward_pre_hooks) == 0` after the context manager exits.
- A run with a zeroing hook on some module changes the loss; the next run without hooks returns loss equal to the first clean run (`torch.allclose`, atol 1e-6). This is the hook-leak test.
- Peak RSS for pythia-70m on the repeated dataset (n=48, half=20) stays under 2 GB (`resource.getrusage`).

## Step 4: experiments

Implement `experiments/induction.py`:
- `prev_token_score(acts) -> (value, bindings)`: for each head, mean attention from position i to i-1; return max and its `(layer, head)`.
- `induction_head_score(acts, min_layer) -> (value, bindings)`: for positions in the second half, mean attention to position `i - half + 1`; max over heads with layer >= min_layer; return value and `(layer, head)`.
- `ablate_head_loss_delta(acts, ctx, layer, head, ablation="zero") -> value`: hook the attention output projection of `layer`, zero the slice for `head`, rerun, return mean loss on the second half (ablated) minus the same from `acts`.

Each experiment takes only `acts` (and `ctx` for interventions) plus declared inputs; no global state.

**Verify:**
- On `pythia-70m` at revision `main`, `induction_head_score` returns a value >= 0.3 and `prev_token_score` >= 0.5 (these are known properties of the trained model). Print the bound heads.
- On revision `step0` (random init), both scores are < 0.15. This is the negative control.
- `ablate_head_loss_delta` on `main` with the bound induction head returns a positive value. Print it. If it's under 0.3, report that and do not change the threshold; we'll discuss.
- Attention patterns are a proper distribution: every row of `acts.attn` sums to 1 within 1e-4 over real tokens.

## Step 5: ledger

Implement `ledger.py`: append-only JSONL at `ledger/results.jsonl` with the result record from SPEC.md; in-memory index on `(claim_id, claim_key)`; `get`, `put`, `stale(claims)` returning claims whose current key has no row.

**Verify:**
- `put` then `get` round-trips every field.
- After writing 3 rows, reopening the ledger from disk yields the same 3 rows.
- `stale()` returns all claims on an empty ledger, none after a full run, and exactly the edited claim plus its dependents after editing one claim's `code.params` (test with the real claim graph: edit `mech.induction_head`, expect it and `causal.induction_head_ablation` stale, `mech.prev_token_head` not).

## Step 6: planner and runner

Implement `planner.py` and `runner.py` per RUNNER.md: group non-stale-skipped claims by pass key, order by checkpoint, topologically sort within a group, one shared pass for observational claims, one additional pass per interventional claim, bindings flow from `exports` to `inputs`, dependents of a failing or non-holding claim recorded as `blocked`. Implement the hook-leak canary (4-sequence loss check after each intervention). Implement `--mode naive` (one forward pass per claim, no sharing) for the throughput baseline. Record `cost` per result.

`claimci plan` prints the plan without loading any model.

**Verify:**
- `claimci plan --revisions main` on a fresh ledger lists 3 claims, 1 shared pass + 1 intervention pass. Running it with no model in the HF cache must still succeed (mock or `--offline` check).
- `claimci verify --revisions main` writes 3 rows; all three hold on `pythia-70m`.
- Rerunning `claimci verify --revisions main` executes 0 claims and reports 3 ledger hits.
- `claimci verify --revisions main --mode naive` produces values equal to shared mode to 6 decimals, and records 3 forward passes instead of 2.
- Force `mech.induction_head` to fail (temporarily set its threshold to 0.99): `causal.induction_head_ablation` is recorded `blocked` with reason naming the parent; no intervention pass runs.
- Make the ablation experiment raise: its row has `status: error`; siblings still complete.
- Inject a hook leak deliberately (register a hook outside the context manager in a test): the canary aborts the group with `hook_leak`.
- Run `claimci verify --revisions main` twice and diff the two sets of values: identical to 6 decimals.

## Step 7: reports

Implement `report.py` and the CLI commands `matrix`, `throughput`, `status`.

- `matrix --question q.induction_formation`: rows = claims, columns = revisions, cell = value with pass/fail mark; markdown to stdout, and `--png` renders the same with matplotlib.
- `throughput`: table with claims verified, forward passes, seconds, claims/minute for shared and naive modes from the ledger's `cost` records.
- `status`: counts by status and evidence code; list of stale claims with the reason (code changed / data changed / never run).

**Verify:**
- After step 6's runs, `matrix` shows one column (`main`) and three rows with correct values.
- `throughput` shows shared forward passes < naive forward passes for the same claims.
- `status` lists zero stale claims; after editing `code.params` of one claim it lists that claim and its dependents, each with reason `code changed`.

## Step 8: the real run

Switch model to `EleutherAI/pythia-160m`. Run:

```
claimci verify --revisions step0,step1000,step2000,step4000,step8000,step16000,step32000,step143000
claimci matrix --question q.induction_formation --png matrix.png
claimci throughput
```

**Verify:**
- Total wall clock under 15 minutes on CPU (print it).
- `step0` column: all three claims fail (negative control).
- Induction head claim starts holding at some checkpoint between step1000 and step16000 and holds at every later checkpoint. If it doesn't, do not adjust thresholds; show me the raw values.
- `matrix.png` exists and renders the table.

## Step 9: README

Write `README.md`: what a claim is (three sentences), the evidence ladder table from SPEC.md, the three CLI commands, the matrix from step 8 inline, and the throughput table. Link SPEC.md and RUNNER.md. No marketing language.

**Verify:** `git clone` into a fresh venv, `pip install -e .`, `claimci validate && claimci plan --revisions main` succeeds from the README instructions alone.

---

Rules for the whole build:
- Every number in a test comes from running the code, never from a guess. If a threshold in the specs doesn't hold on the real model, report the measured value and stop; don't tune the threshold to pass.
- Keep each module under 250 lines. No frameworks beyond what's listed.
- Commit after each step with the step name in the message.
