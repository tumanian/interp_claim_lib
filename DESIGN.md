# claimci design: from v0 to a knowledge base for research agents

Status: proposal, not built. SPEC.md and RUNNER.md describe what exists today.
This document describes where it goes next and why. Where they disagree, this
document is the intent and the specs get updated as each piece lands.

## 1. Who this is for

The main user is a research agent working on interpretability. Its loop:

1. It has an idea: a hypothesis about a model, a component, or a mechanism.
2. It scans prior work to reason about the idea: what is known, what is only
   asserted, what it can build on, what would be new.
3. Sometimes it tests the idea, and the result goes back into the base, linked
   to the prior work it touched.

Most traffic is reading (step 2). Testing (step 3) is the occasional write. A
nightly job keeps the base fresh: new papers, new models, changed code.

v0 was built around one paper reproduced on one model, with a matrix for
humans. That remains useful, but it is a special case of the above.

### Worked example

Idea: "In pythia-160m the induction head also does fuzzy copying: it matches
`cat` to `cats`, not only exact repeats."

A scan should return, and make impossible to misread:

- `mech.induction_head`: measured on pythia-160m, exact repeats of random
  tokens, bound to L4H6, observational (IOB), 0.878 against a threshold of 0.3
  that we chose.
- `causal.induction_head_ablation`: measured, causal (IMP), zeroing one head,
  +0.92 nats.
- `lit.olsson.fuzzy_copying`: asserted by the paper, not tested on any model
  here.

From that the agent can reason: L4H6 is an established exact-match induction
head with causal evidence; fuzzy copying is a paper assertion with no test; the
idea is a Pythia test of that assertion; it can reuse the L4H6 binding and needs
only a near-repeat dataset. It writes a new test with
`about: lit.olsson.fuzzy_copying` and `depends_on: [mech.induction_head]`.

## 2. Three kinds of claim

The paper's spirit and the exact values are different objects. If a model forms
induction heads abruptly at step 300 and another at step 10000, both support
"induction heads form in an abrupt phase change early in training". The step
number is a property of the model and its training run, not a test of the paper.

| | `paper` | `test` | `measurement` |
|---|---|---|---|
| is | what the authors assert | a pass/fail check, usually of a paper claim | a recorded number |
| evidence | TAS / NAS | rerunnable (IOB, ICP, IMP, ISS, ICA) | rerunnable |
| code + data | none | required | required |
| `expect` | none | required | forbidden |
| `about:` | n/a | optional, a `paper` claim | optional, a `paper` claim |
| status | computed from linked tests (section 5) | proposed, verified, refuted | proposed, measured |

A test is a measurement plus an `expect`. Today's `mech.induction_head` is the
induction score (a measurement) with a 0.3 threshold attached (a test). The
`kind` field makes that explicit and `validate` enforces the table.

Paper claims never become verified. The current SPEC path where a TAS claim with
an attached experiment is "promoted" to IMP goes away. Reproducing a number the
paper states becomes a `test` whose `expect` is stated by the paper (section 3).

A second, independent field says what a test or measurement reads:

- `scope: checkpoint` (default): one model at one revision. Everything today.
- `scope: trajectory`: another claim's values across a list of revisions
  (`over: {claim, revisions}`). No forward pass; it reads the ledger. Examples:
  "the induction score rises abruptly" (test), "the onset step" (measurement).

All kinds stay in `claims/` with one loader, so `depends_on`, `about` and the
DAG checks work across kinds.

### Examples

```yaml
id: lit.olsson.phase_change
kind: paper
evidence: TAS
statement: >
  Induction heads form abruptly, in a phase change early in training,
  coinciding with a jump in in-context learning.
source_scope: small attention-only and standard transformers trained by the authors
provenance:
  source: {kind: paper, ref: arXiv:2209.11895, locator: "<from the paper>"}
  assigned_by: agent:ingest
```

```yaml
id: dev.induction_abrupt
kind: test
scope: trajectory
about: lit.olsson.phase_change
evidence: IOB
statement: At least 80% of the final induction score arrives within one doubling of training steps.
over: {claim: mech.induction_head, revisions: [step0, step1, step2, ..., step143000]}
code: {entrypoint: claimci.experiments.trajectory:max_rise_per_doubling, version: local}
metric: fraction_of_final_in_one_doubling
expect: {op: ">=", value: 0.8}
control: {revisions: [step0, step1, step2, step4], expect: {op: "<", value: 0.3}}
provenance:
  supplemented:
    expect: {by: ..., note: the paper says abrupt but gives no number}
    over.revisions: {by: ..., note: every checkpoint to step1000, then a sparse sample}
```

```yaml
id: dev.induction_onset_step
kind: measurement
scope: trajectory
about: lit.olsson.phase_change
evidence: IOB
over: {claim: mech.induction_head, revisions: [...]}
code: {entrypoint: claimci.experiments.trajectory:onset, params: {threshold: 0.3}}
metric: onset_step
provenance:
  supplemented:
    code.params.threshold: {by: ..., note: same threshold as mech.induction_head}
```

## 3. Every number says where it came from

Every number in a test or measurement (`expect`, thresholds in `code.params`,
checkpoint lists, dataset parameters that encode a judgement) is either:

- stated: from the source, under `provenance.source` with a locator; or
- supplemented: ours, under `provenance.supplemented.<field path>` with who set
  it and why.

`validate` rejects a claim with an untagged judgement number. Today only
`expect` is covered (`provenance.supplemented.expect`); this widens it to any
field path.

A verdict records its basis: `stated_value` (a number from the source
reproduced within tolerance) or `supplemented_expect` (holds under our
threshold). This exists today for promotion and extends to every verdict.

## 4. Results

Changes to what the runner records:

- Values are always recorded. A claim is computed whenever its inputs exist.
  "Blocked" becomes derived at read time: an ancestor does not hold under the
  current thresholds. This matches rule 5 (editing `expect` re-evaluates without
  rerunning), lets the matrix show `0.085 (gated)` instead of hiding the onset,
  and gives trajectory claims a full series. Dependents of an ancestor that
  errored or produced no bindings are still recorded as blocked, with no value.
- Margin and sensitivity are computed from the stored value and the current
  `expect`: distance to the threshold, and the threshold range over which the
  verdict would flip. An agent should lean on 0.878 against 0.3, not 0.31 against 0.3.
- Pre-registration: the ledger records `expect` with the first result for a
  claim key. If `expect` is edited after results exist, verdicts carry
  `post_hoc: true`. Editing stays allowed; it just stops being invisible.
- Controls: a test may declare `control:` (revisions, a dataset, or a random
  component) with its own `expect` that must hold, typically "fails on step0".
  A verdict whose control has not run or did not behave is reported as
  unconfirmed. Today step0 is only a convention in `check_step8.py`.
- Environment fingerprint: torch, transformers and claimci versions on every
  row. It is not part of the claim key (that would invalidate everything on each
  upgrade); drift is caught by the weekly sample in section 7.

## 5. Read path: what an agent calls

The agent interface is queries, not YAML. Every read command takes `--json`.

| command | returns |
|---|---|
| `claimci search <text> [--model M] [--component L4H6] [--kind ...]` | matching claim cards, ranked |
| `claimci card <claim> [--model M]` | one claim card |
| `claimci explain <paper ref or paper claim>` | each paper claim with its verdict, linked tests and measurements, gaps |
| `claimci get <claim> --model M --rev R` | one result: value, verdict, margin, bindings, control, post_hoc, stale |
| `claimci gaps [--budget N]` | untested or partly tested paper claims, with the cheapest next step and its cost |

### The claim card

A short, honest summary sized for an agent's context. It must make "asserted by
the paper" and "measured on this model" impossible to confuse, because that is
the most likely reasoning error. A sketch:

```
mech.induction_head  [test, IOB, checkpoint]  about: lit.olsson.induction_heads_exist
  measures: mean attention from a repeated token to the token after its earlier occurrence
  data: repeated_random_tokens (n=48, half=20), synthetic, not natural text
  threshold: >= 0.3 (supplemented: set for claimci v0, the paper gives none)
  results:  pythia-160m@main  0.878 verified  margin +0.578  binds L4H6  control step0 0.037 ok
            pythia-160m 19 checkpoints: holds from step1000 on
  not shown: copying (OV circuit); natural text; other models
```

### Paper-claim verdicts

Computed at read time, never stored, and per model:

```
lit.olsson.phase_change
  tests:        dev.induction_abrupt   verified on pythia-160m
  measurements: dev.induction_onset_step = step1000 on pythia-160m (step512 before)
  gaps:         nothing tests the coincidence with in-context learning
```

A failed test shows as "refuted by X on model M". The system does not conclude
that the paper is wrong; the test may be a poor operationalization. That is a
judgement for a person or a reviewing agent.

### Links

`depends_on` stays computational (inputs and bindings flow along it). Meaning
gets its own fields, used by search and cards:

- `about:` a test or measurement to the paper claim it bears on
- `relation:` between paper claims: `same_as`, `refines`, `contradicts`

## 6. Write path: a new idea or an old claim on a new model

1. The agent writes a claim file (or `gaps` proposes one).
2. `validate`: kinds, number provenance, links, DAG.
3. `verify`: the runner as today. Thresholds are pre-registered by the first run.
4. The result is linked to prior work through `about` and `depends_on`.

Running an existing test on another model should not mean copying the file.
`verify --models` (like `--revisions`) runs a claim on each listed model, with
the model in the claim key as it is today. Results are per model; the card
shows them side by side.

A failure on a new model is information about scope, not a refutation: "holds
on 160m, not on 1b". Paper-claim verdicts are therefore per model ("3 of 4
models"), never one global yes/no.

## 7. Nightly refresh

```
 new papers ───────► 1. ingest ──► paper claims (proposed, machine-made)
                                        │
 new models, ──────► 2. expand ◄────────┘  tests x registry models not yet run
 new checkpoints                            + untested paper claims with a cheap test
                          │
 code / data edits ──► 3. plan ──► priority + budget ──► 4. verify ──► 5. diff ──► morning report
```

1. Ingest: fetch new papers in scope; extract `paper` claims with locators; link
   them to existing paper claims (`same_as`, `refines`, `contradicts`). This is
   the LLM-heavy step.
2. Expand: for each test, the compatible models in the registry it has not run
   on, and new checkpoints of tracked models.
3. Plan: tests x models x checkpoints grows fast, so the planner takes a
   compute budget and a priority order:
   1. stale claims (code or data changed)
   2. claims whose verdict flipped since the last run
   3. paper claims with no test, where a cheap test exists
   4. tests run on one model, extended to the nearest model in the registry
4. Verify: the existing runner. Ledger keys make it idempotent.
5. Diff: new paper claims, new verdicts, verdicts that changed and why
   (code, data, threshold, model, environment), and results that disagree across
   models.

Weekly: rerun a random sample of claims with `--force` and compare values. If
values moved without a key change, report the environment difference (section 4).

### Trust gate

Ingested paper claims and auto-proposed tests land as `proposed` with
`assigned_by: agent:...` and stay marked machine-made until reviewed by a person
or a reviewing agent (`reviewed_by:`). Cards show the mark. Auto-generated tests
must have their thresholds fixed before the first run. Without this, one bad
extraction becomes tomorrow's prior work.

### Model registry

`models.yaml`: model id, revisions to track, adapter, and a compatibility tag
(architecture family). A test runs on a model only if its experiment supports
that adapter. Today only the HF GPT-NeoX adapter exists, which covers Pythia
70m to 12b; Llama-style and OLMo models need adapters for module names (the
ablation hooks the attention output projection by name).

### Where it runs

Pythia up to about 1b runs on CPU; the full 19-checkpoint run of pythia-160m
takes under a minute once cached. One scheduled box (for example EC2) covers
verify and diff. Ingest needs an LLM and can run as a scheduled agent.

## 8. Changes to existing behaviour

| today | after |
|---|---|
| dependents of a non-holding claim recorded `blocked`, no value | value recorded; gating derived at read time |
| TAS claim + experiment promoted to IMP | paper claims never verified; tests with a stated `expect` reproduce numbers |
| `provenance.supplemented.expect` only | `provenance.supplemented.<any field path>` |
| one model per claim file | `verify --models`; results per model |
| `check_step8.py` encodes the step0 control | `control:` on the test |
| no environment on results | fingerprint on every row; weekly drift sample |
| human reports only (matrix, throughput, status) | plus `search`, `card`, `explain`, `get`, `gaps` with `--json` |

The current three claim files become `kind: test` with
`about: lit.olsson.induction_heads_exist` (prev-token and induction) and
`about: lit.olsson.ablation_removes_icl` (ablation).

## 9. Build order

Each step ends with something usable.

1. Olsson et al. as paper claims: fetch the paper and write 8 to 12 `paper`
   claims with real locators and wording (what is in this document about the
   paper so far is from memory). Add `kind` and `about` to the loader and
   `validate`. Migrate the three existing claims.
2. Results: always record values, derive gating, add margin, sensitivity and
   environment fingerprint.
3. Read path: `card`, `explain`, `get`, `search`, `gaps`, each with `--json`.
   Check it by running the fuzzy-copying example by hand.
4. Trajectory scope: `dev.induction_abrupt` and `dev.induction_onset_step`.
5. Pre-registration and `control:`.
6. Nightly without ingestion: model registry (pythia 70m, 160m, 410m, 1b),
   `verify --models`, expand, plan with budget, diff report, weekly drift sample.
   This also answers whether the onset step moves with model size.
7. Ingestion with the trust gate.

## 10. Open questions

- Search: plain text and tag matching first, or embeddings from the start?
- Granularity of paper claims: how finely to split a paper (one claim per
  figure, or per argument)? Too coarse and tests cannot target them; too fine
  and the base is noise.
- Who reviews machine-made claims, and what does review check?
- Controls for trajectory tests: is step0 enough, or should every test also run
  on a shuffled-data or random-component baseline?
- Copying score: the paper's definition of an induction head includes copying
  (OV circuit). Add a measurement for it before calling
  `lit.olsson.induction_heads_exist` supported, or record it as a gap?
