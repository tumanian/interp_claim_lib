# claimci design: from v0 to a knowledge base for research agents

Status: proposal, not built. SPEC.md and RUNNER.md describe what exists today.
This document describes where it goes next and why. Where they disagree, this
document is the intent and the specs get updated as each piece lands.

## 0. Motivation and architecture (brainstorm, to prune)

This section is brainstorm: the larger shape the rest of the design could grow
into. Sections 1 to 10 are the near-term plan. Expect to cut from here.

### Why this works for interpretability and not for genomics

The evidence ladder is borrowed from Gene Ontology annotation codes. In
genomics, "IMP: inferred from mutant phenotype" records that someone once
knocked out a gene in a lab. The annotation is a static label for an experiment
nobody will rerun. In interpretability the "mutant" is a forward pass with a
hook, so the evidence code is earned each time the code runs.

| | genomics | interpretability |
|---|---|---|
| cost of an experiment | weeks, money, a lab | seconds to hours of compute |
| replication | rare, a project of its own | a rerun; can happen every night |
| the subject | living organisms, variable | a fixed checkpoint; reruns are bitwise identical on one machine |
| generalization | another species is another project | another model is another ledger row |
| development | sampled when someone pays for it | Pythia and OLMo publish every stage of training |
| the expensive part | running the experiment | deciding what the experiment should be |

So the curation model carries over (claims, evidence codes, a base agents can
read), and verification can be continuous. The cost moves rather than
disappearing: compute is cheap up to a point (a 70B model or training an SAE is
not), and judgement about how well a test captures a claim becomes the limit.

### What can be verified automatically

Connected end to end, ingest, tests and the nightly job make an automatic
verifier of the literature: paper, paper claims, tests, runs on open models,
per-model verdicts, morning report. Its quality is the quality of its
auto-written tests.

| step | automatable | why |
|---|---|---|
| ingest | mostly | quotes and numbers can be string-matched against the paper |
| writing the test (spirit to exact values) | the hard step | which metric, data, threshold and control capture the authors' claim |
| run, record, diff | fully | exists |
| reading a verdict | partly | "refuted" can mean the paper is wrong or the test is bad |

Principles that follow:

- Several independent tests per paper claim. Agreement makes the verdict about
  the paper; disagreement points at the tests, and the report should say so.
  This sits next to "check across models".
- Pre-registration and controls are required for auto-written tests, or the
  system drifts toward tests that pass.
- Review auto-written tests, not every extraction. The share of auto-verdicts
  that survive review is the health metric for the whole system.
- The experiment library sets coverage. Today: attention patterns and head
  ablation. Next: activation patching, probes, steering, SAE features. Running a
  paper's own code through an adapter is the shortcut.
- Reachable: open-weight models, especially those with public training
  checkpoints. Out of reach: closed models, claims that need large compute,
  claims whose evidence is qualitative case studies.

### Local-first: verify locally, push to a shared base

claimci runs on anyone's machine; a shared base (a git repo of claims and ledger
rows) collects what people push. The name already fits: CI for claims.

```
local                                             shared base (git repo)
pull  <──────────── prior work ─────────────────  claims/, ledger/
write a claim or idea
claimci plan  -> "~40 GPU-min on a cloud backend, ~$3"
verify (CPU, or pay for a GPU backend)
push  ─────────── PR: claim file + ledger rows ──> CI reruns what it can afford
                                                   -> merged with a trust level
```

Trust comes from reproduction, not reputation:

| trust level | meaning |
|---|---|
| reported | someone ran it and pushed it; nobody else has rerun it |
| CI-reproduced | the shared CI reran it: same value within tolerance, same verdict |
| independently reproduced (N) | N other parties reran it on their own hardware |

Expensive results stay "reported" until someone pays to reproduce them, and the
card says so.

What local-first needs beyond sections 1 to 10:

- Pinned code. `code.version: local` hashes a module on one disk; a pushed claim
  must reference a repo and commit (`git:<sha>`) so others can rerun it.
- Agreement within tolerance. Same-machine reruns are bitwise identical; CPU vs
  GPU and different BLAS builds are not. Reproduction means values within a
  tolerance and the same verdict, with the environment fingerprint recorded.
- Priced backends (P2: only matters past what a CPU can run). The Backend
  interface already separates where from what. A cloud GPU backend plus a cost
  per pass lets `plan` show money; the user approves the spend, then it runs.
- Registration before results. Locally, "threshold set before the run" is only
  as good as the local clock. In a shared base the claim is pushed first (test
  and threshold fixed), then the results, as in a clinical-trial registry.
  Optional, but cards show whether a result was registered first.
- Mergeable ledger. It is already append-only JSONL keyed by content hashes;
  rows from different people for one key are more evidence, and disagreements
  between them go to the diff report.

The nightly job (section 7) becomes the shared base's CI. A researcher's local
agent pulls the base, reasons over it (section 5), tests an idea locally, and
pushes.

### Volunteer computing: lessons from SETI@home / BOINC

The local-first shape is close to SETI@home and Folding@home (both on BOINC),
which already met several of these problems.

- Redundancy is the trust model. BOINC sent each work unit to two or more
  volunteers and accepted a result only on a matching quorum, because hardware
  was faulty and some people cheated for credit. That is "independently
  reproduced (N)", and BOINC's validator is the shared CI's job.
- Floating point across machines. BOINC used homogeneous redundancy (compare
  only within a platform class) and fuzzy validators (compare within a
  tolerance). We need both: exact matching within a hardware class, tolerance
  across classes.
- Work sized to the volunteer. Pythia on CPU fits a laptop; a 70B model does
  not. The planner tags work by hardware class.
- Incentives get gamed. Credit should go to reproductions that others confirm,
  not raw submissions.

Where it differs:

- Contributors bring ideas as well as compute: closer to BOINC plus Wikipedia.
  The judgement-heavy part (writing good tests) is distributed too.
- Work is content-addressed: the same claim, model and checkpoint give the same
  key, so duplicates are free.
- Every result is useful: a failure on a new model is still scope information.
- `claimci gaps` is the work queue volunteers pull from: reproduce a reported
  result, extend a test to another model, run a missing control.

The new risk: verifying a pulled claim runs someone else's code. BOINC avoided
this by running only the project's signed applications. Here, by default,
volunteers run only experiments from a reviewed library at signed commits;
arbitrary pushed code runs in a sandbox or only on CI until reviewed. This has
to be in the design from the start.

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

- Tools layer: only the judge is a standalone tool (`claimci judge`). Re-judging, steering
  diagnostics and data vendoring are scripts; the quote checker, source fetching, model helpers,
  interventions, readouts, the remote runner and the artifact store are buried in the pipeline.
  Should they become `claimci tools ...` (judge, check, fetch, vendor, probe, compare-judges,
  run-remote) with JSON output, built on the same pieces as experiments (section 16), so what a
  person or agent sees while exploring is what a claim later tests?

## 11. Demo: drop an arXiv id, get verified claims (brainstorm)

A small web app that shows the whole loop on one screen.

```
[ arXiv id ]
 1. Read     fetch the paper (arXiv HTML or PDF) -> text with section anchors
 2. Extract  LLM -> paper claims (statement, quote, locator, scope, hedge, stated numbers)
             + references: code repos, datasets, models (HF ids), checkpoints
             each quote and number string-checked against the paper text, shown as pass/fail
 3. Match    per claim: a test or measurement in the library? the paper's own code? neither -> gap
 4. Plan     claims x models x checkpoints, estimated time (and cost, once backends are priced)
 5. Verify   runs; results stream into claim cards: value, verdict, margin, control
 6. Write    claim files, ledger rows and card metadata (download, or open a PR to the shared base)
```

The moment the demo is built around: a paper claim card goes from "asserted"
to "measured on pythia-160m: holds, margin +0.58, control ok" seconds after the
id is dropped.

Step 3 decides how good it looks, so the demo is honest about it:

- Golden path, Olsson et al.: the library covers it (previous-token head,
  induction head, ablation, onset and abruptness). End to end in about a minute
  on CPU.
- Any other paper: reading, extraction, quote checks, references and the plan
  work for anything. Verification runs only where the library has a matching
  experiment; otherwise the card says "no test available" and what one would
  need. Showing gaps is part of the trust model, not a failure of the demo.
- The paper's own code: "code found, not runnable yet" until an adapter exists.
  Stretch goal.

### Steering papers

Steering is the second demo family, because it is popular, cheap to rerun on
small models, and easy to overclaim. Its evidence code is ISS (inferred from
steering): add a direction, or scale a set of neurons, and the predicted
behaviour appears.

Candidate papers to check (recent, from a search; not yet chosen):
- Query-Adaptive Neuron Steering for LLMs (arXiv:2610.14519): a trained gate
  sets per-query coefficients on a candidate set of neurons; claims gains across
  model scales and transfer across benchmarks.
- Does Steering Break Your Model? (arXiv:2610.07722): SteerScope, 23 methods
  and 15 metrics; reports that no activation steering method beats prompt
  steering on efficacy against side effects, and that side effects grow out of
  distribution.

The second paper is close to a checklist for what a steering test needs. A
steering claim is verified as a bundle of tests and measurements, not one
number:

| piece | kind | what it checks |
|---|---|---|
| efficacy | test | behaviour metric, steered minus unsteered, on held-out prompts |
| dose-response | measurement | effect against coefficient; a real effect should be monotone over a range |
| specificity control | test | the same number of random neurons (or a random direction of the same norm) in the same layer, same coefficient, gives a much smaller effect |
| side effects | measurement | perplexity on held-out natural text and a small capability set, steered vs unsteered |
| prompt baseline | measurement | the same behaviour asked for in the prompt, no activation change |
| held-out selection | rule | neurons or directions chosen on one prompt set, evaluated on another |
| across models | expand | rerun on other sizes when the method claims to be general |

Library additions this needs:
- a steering hook: add a vector to the residual stream, or scale chosen MLP
  neurons, at a layer, with a coefficient; reuses the hook context manager and
  the hook-leak canary;
- generation-free metrics where possible (log-probability of target tokens),
  because they are deterministic;
- when a judge model scores behaviour, the judge's id and version go into the
  claim key, and the evidence is marked as judged, since judges drift.

Compute: steering papers usually report 7B to 8B models. Small models (Pythia,
small Llama-style) run on CPU and are enough to check whether the method's
effect exists and passes its controls. Matching the paper's own model needs a
GPU, which is where priced backends (section 0, P2) stop being P2.

### Claims from posts: "steer two neurons and you get evil Qwen"

Much of what an interpretability agent will meet is not in papers. The
motivating example is a post on X (@_lyraaaa_, 2026-10-23, about 3.9M views):

> if you have a qwen3.5 4b around, please try steering neurons L14 N8949 +1
> and L13 N7956 -1 / now you have evil qwen / recipe: dose × median(‖resid‖
> leaving layer L, tokens 1:) / ‖W_down[:, n]‖ at the down_proj at every input

Posts like this are why `kind: paper` should become `kind: source`, with
`provenance.source.kind: paper | post | blog | neuronpedia | ...`.

It splits cleanly along the spirit/values line:

- The intervention is stated exactly: model, two neurons, signed doses, and a
  formula for the activation added at the input of `down_proj` at every
  position. These are stated values (section 3), not ours.
- The behaviour is not: "evil" has no metric. How to measure it is the
  supplemented judgement, and the verdict depends on it.

```yaml
id: src.lyra.evil_qwen
kind: source
evidence: NAS
statement: >
  In Qwen3.5-4B, adding to MLP neuron 8949 of layer 14 (dose +1) and neuron 7956
  of layer 13 (dose -1), scaled by the recipe, makes the model behave evilly.
quote: "if you have a qwen3.5 4b around, please try steering neurons L14 N8949 +1 and L13 N7956 -1"
hedge: shows
subject: {model: {hf: <Qwen3.5-4B repo id>}}
stated:
  intervention:
    - {layer: 14, neuron: 8949, dose: +1}
    - {layer: 13, neuron: 7956, dose: -1}
  activation_added: dose * median(norm(resid after layer L), tokens 1:) / norm(W_down[:, n])
  where: down_proj input, every position
provenance:
  source: {kind: post, ref: "https://x.com/_lyraaaa_/status/2113833330429083662", date: 2026-10-23}
  supplemented:
    layer_indexing: 0-indexed decoder layers (the post does not say)
    median_scope: per input, over its own tokens excluding the first
    behaviour_metric: see the tests below
```

One property of the recipe matters for verification: at dose 1 the added
vector has the norm of a typical residual stream vector, which is a large
perturbation. Large perturbations make models strange with almost any
direction, so the random-neuron control below is the most important test.

| piece | kind, evidence | what it checks |
|---|---|---|
| efficacy, multiple choice | test, ISS | on persona-style multiple-choice questions (power seeking, harming humans, deception), the log-probability of the misaligned option rises. Deterministic, no judge |
| efficacy, open-ended | measurement, ISS (judged) | generations on fixed open-ended prompts scored by a judge model for misalignment; judge id and version in the claim key |
| random-neuron control | test | two random neurons at layers 13 and 14 with the same recipe and doses (same perturbation norm) do not raise the misalignment score. The claim stands or falls on this |
| each neuron alone | measurement | +1 on L14 N8949 only, -1 on L13 N7956 only: which one carries the effect |
| dose-response | measurement | doses 0, 0.25, 0.5, 1, 2 |
| coherence | measurement | perplexity on natural text and a small capability set, steered vs not: evil, or just broken? |
| prompt baseline | measurement | "you are an evil assistant" as a system prompt, no steering |

Compute: about 4B parameters is about 8 GB in bf16, which fits on a 16 GB
laptop. Log-probability tests are cheap; open-ended generation on CPU is slow
but feasible for a few dozen prompts. This one can run locally.

### Another post: "Qwen3-8B has a Queen Elizabeth II neuron"

A post on X (@_a9lim, 2026-10-26): "qwen 3 8b appears to have a queen
elizabeth the second neuron (L11N3799)". Text only: no code, no data, no method
stated. A concept-neuron claim rather than a steering claim: precise about the
component, silent about the evidence, hedged ("appears to").

```yaml
id: src.a9lim.qwen3_8b_qe2_neuron
kind: source
evidence: NAS                     # no method or figure to point to
statement: Qwen3-8B has a neuron (layer 11, neuron 3799) that represents Queen Elizabeth II.
quote: "qwen 3 8b appears to have a queen elizabeth the second neuron (L11N3799)"
hedge: suggests
subject: {model: {hf: Qwen/Qwen3-8B}, component: {layer: 11, neuron: 3799}}
provenance:
  source: {kind: post, ref: "https://x.com/_a9lim/status/2114801107931185322", date: 2026-10-26}
  supplemented:
    subject.component: >
      "L11N3799" read as 0-indexed layer 11, index 3799 of the MLP hidden
      activation (the input to down_proj). The post does not say.
```

Resolving "L11N3799" is itself a judgement (0- or 1-indexed layer; which
activation counts as the neuron). It is recorded as supplemented and fixed
before any run. Trying both readings and keeping the one that works is a forking
path; if both are run, both results are kept.

"X is a Y neuron" breaks down into tests at increasing strength. Each one
narrows what the neuron is, so a verdict can say "a British-monarchy neuron,
not specific to Elizabeth II" rather than just pass or fail:

| piece | kind, evidence | what it checks |
|---|---|---|
| selectivity | test, IOB | activation on Elizabeth II text vs matched controls: other monarchs (Victoria, Elizabeth I, Charles III), other famous women, other Elizabeths, other senses of "queen" (chess, the band). AUROC per control class |
| top activations | measurement, IOB | share of the neuron's top-k contexts in a natural-text sample (e.g. a FineWeb slice) that are about Elizabeth II; polysemanticity check |
| steering | test, ISS | clamping or amplifying it on neutral prompts raises the log-probability of Elizabeth II tokens; dose-response; random-neuron control in the same layer; side effects |
| ablation | test, IMP | zeroing it hurts Elizabeth II facts (log-probability of correct completions) more than facts about other monarchs |
| robustness | measurement | paraphrases and other languages |

Thresholds for all of these are supplemented: the post states no numbers.

Compute: Qwen3-8B in bf16 is about 16 GB of weights, more than a 16 GB laptop
can hold with activations. Quantizing changes activations, which is a problem
when the claim is about one neuron's values. So this needs a GPU (a single
24 GB card is enough), which moves priced backends from P2 to P1 as soon as the
demo includes a model of this size.

### Build order for the demo

The demo reorders section 9: ingestion moves first, the nightly job later.

1. Ingestion: Claude API extraction into `kind: paper` claims, plus the quote
   and number check.
2. `kind`, `about` and the claim card (section 9 steps 1 and 3): the cards are
   the UI.
3. Recorded values and margins (step 2), so cards show numbers, not "blocked".
4. Trajectory tests (step 4), for the onset result.
5. Steering library: hook, random-neuron control, side-effect and dose-response
   measurements, on a small model.
6. A thin web layer: a small Python server wrapping claimci and one page that
   streams results. Locally first; a Hugging Face Space on CPU for a shareable
   link.

Open: which golden paths besides Olsson. Candidates: the evil-Qwen steering
post (runs locally), the Qwen3-8B Elizabeth II neuron post (needs a GPU), and
a steering paper.

## 12. Objects, annotations, mechanisms

Claims are about different kinds of things, and the read path should say which.

| layer | what it is | examples | bio analogue |
|---|---|---|---|
| object | a component with an identity in one model | neuron L14N8949 of Qwen3.5-4B; head L4H6 of pythia-160m; the refusal direction of gemma-2b-it (a saved vector); an SAE feature | a gene or variant (Ensembl, dbSNP) |
| annotation | a claim that one object does something | "L4H6 is an induction head"; "L11N3799 is an Elizabeth II neuron"; "these two neurons push Qwen toward evil" | a GO or ClinVar annotation: variant to function, with an evidence code |
| mechanism | a claim about how a computation works, across objects | "refusal is mediated by a single direction"; "induction is a previous-token head plus an induction head"; "induction heads form in a phase change" | a pathway (Reactome) |

Objects are anchors, not claims: nothing about an object is verified except through the
annotations and mechanisms that reference it. Their job is aggregation: everything checked
about one component, across sources, in one place.

Object ids: `<model>@<revision>/<kind>/<locator>`, e.g.
`Qwen/Qwen3.5-4B@main/mlp_neuron/L14/N8949`, `EleutherAI/pythia-160m@main/attn_head/L4/H6`,
`google/gemma-2b-it@main/direction/<artifact id>`. They are resolved from what claims already
carry, so nothing new is maintained by hand:
- `subject.component` and stated interventions (layer, neuron) on a claim;
- exported bindings (`layer`, `head`) of a result: the induction claim's bound head becomes an
  object at each revision where it was found;
- direction artifacts exported by a claim.

Findings get a `level: annotation | mechanism` (ingest proposes it; review confirms). A
mechanism usually depends on annotations ("a single direction mediates refusal" needs the
direction object and its ablation/addition results first).

Relation to Neuronpedia: Neuronpedia is an object atlas with mostly automated annotations
(auto-interp labels, activation examples, steering on demand). In our terms that is
*Automated | Auto-interp label* and *Correlational | Activation reading* evidence. claimci is
the verification layer: annotations and mechanisms as checkable claims with controls. The two
are complementary; ingesting Neuronpedia labels as stated annotations is deferred.

Build: object resolution and an Objects list on the report card first; per-object pages
(everything known about one component) next; `level` on findings with the next ingest change.

## 13. V2: the app writes its own experiments (self-improving, safely)

Today a finding is verifiable only if the experiment library already has code for it: matching
finds a template or reports a gap. Emergent Misalignment (arXiv:2502.17424) is the example: all
seven findings are gaps because every one needs fine-tuning, and the library only does
inference-time work. In V2, a gap becomes a task: the app writes the experiment, proves it
correct, and adds it to the library.

The loop:
1. A gap names what is missing (here: fine-tune a model on a dataset, then judge it).
2. An agent writes the experiment as library code: an entrypoint, its template, and tests.
3. The code runs in a sandbox (no credentials, network only to the model and dataset hosts it
   declared, CPU/GPU and time budgets), never on the box that holds the ledger or keys.
4. Before it can verify anything new, it must reproduce something already known: a number the
   authors state, the authors' own artifact, or an existing verified claim (as the refusal
   direction reproduced the published vector at cosine 0.99999). Code that cannot reproduce a
   known result does not get to judge an unknown one.
5. A reviewer (a person, or a reviewing agent with a person signing off) reads the diff and the
   reproduction result, then the code is merged into the library and versioned like any other.
6. Everything it produces is labelled: on report cards, code shows as "made here: generated by
   an agent, reviewed by X", next to "written in the library" and "pulled from the authors".

Safety rules:
- Generated code never sees credentials. The judge, model downloads and dataset pulls happen
  through narrow, audited calls the sandbox requests; it cannot reach the AWS, HF or Anthropic
  keys.
- Budgets are part of the task: GPU-hours and API dollars, set before the run, enforced by the
  runner, shown on the card.
- No self-modification of the core: an agent may add experiments, templates and data; the
  runner, ledger, validation, judge and safety checks are changed only by people.
- Pre-registration applies to generated tests exactly as to written ones: thresholds are fixed
  before the first run, and a generated test that changes its own threshold is flagged post hoc.
- Promotion is reversible: a generated experiment found wrong later is retired, and every claim
  it touched goes stale.

First V2 target: a fine-tune-then-judge experiment (LoRA on the authors' datasets, judged with
the existing judge), checked first by reproducing the paper's own controls (secure-code and
educational fine-tunes stay aligned) before testing its headline claim.

## 14. Decouple stored data from code layout (built)

Today two things tie the ledger to how the code is laid out:
- claim files name a Python path (`entrypoint: claimci.experiments.direction:steer_ab`);
- `code_version` hashes the whole module file, so moving a file, changing an import or a comment
  makes results stale, while a change to a helper in another module that the experiment calls is
  missed.

Plan:
- Stable experiment ids, registered where the function lives: `@experiment("caa.steer")`. Claim
  files name the id, never a path. The registry doubles as the experiment catalog for matching,
  `expand`, and V2's agent-written experiments.
- Behaviour hash: the normalized syntax tree of the function plus every in-package function it
  calls, transitively; no file paths, import lines, comments or docstrings. External code stays
  pinned by commit (`git:<sha>`).
- Datasets and artifacts are already keyed by content; models by id and revision.
- Migration: rewrite entrypoints to ids; for each result whose behaviour hash is unchanged, append a
  copy under the new key with `migrated_from: <old key>` (append-only, no reruns). Check: no claim
  goes stale; tests pass.

## 15. Group the package by concern (built)

```
claimci/
  schema/       claim.py, repo.py, codehash.py, source_tests.py, loader.py, validate.py, evidence.yaml
  store/        ledger.py (+ current_row, gated_by), artifacts.py, datasets.py
  run/          planner.py, runner.py, verify.py, backends.py, models.py, formats.py, operations.py,
                interventions.py, readouts.py, adapters.py, executors.py
  experiments/  the experiment library (one module per family)
  sources/      arxiv.py, web.py, post.py, document.py, extract.py, match.py, pipeline.py,
                expand.py, templates.py
  read/         cards.py, report.py, objects.py, agent.py, judging.py
  app/          web.py, static/index.html
  cli.py
```

Dependencies point one way: schema <- store <- run <- experiments; sources on top; read uses
schema and store (and the judge from experiments, for the judge tool); app and cli may import
anything. tests/test_layout.py checks every relative import, including ones inside functions, and
that the top level holds only entry points. One named exception: schema/validate.py reads
store/datasets.py to check that data files hash as declared.

Two moves came out of the layering rather than the plan: `verify()` lived in the CLI and the local
executor imported it from there (run -> cli), so it is now run/verify.py; and gating (`gated_by`,
`current_row`) is a read of the ledger, so it lives in store/ledger.py, which run can use for
promotion without importing read/.

Not done: templates stay one module with packs in code (they are small), and the CLI stays one
module. Split either when it grows.

Claim keys: 100 of 106 claim@revision keys were identical before and after. The other six (the
gemma refusal family) changed because `DATA_DIR = Path(__file__).resolve().parents[1] / "data"`
became `parents[2]` when datasets.py moved, and the behaviour hash hashed that expression. A
constant derived from `__file__` is a location, not behaviour, so the hash now records it as
`<location>`; their ledger rows were carried over with `migrated_from` (scripts/migrate_keys.py),
nothing recomputed. Further moves change no key.

## 16. Backends, executors, and experiments as compositions (built: torch backend, executors, formats, operations, interventions, readouts; not yet: other backends, batching across claims)

`CpuBackend` is a v0 name. The Pythia path does run on CPU in float32, but chat models load
through `chat.load_chat`, which picks CUDA when available, so the GPU runs went through a class
called `CpuBackend`. Worse, the `Backend` interface (load / run / probe / unload) only covers the
original shared-forward-pass case: the refusal, steering, direction and judge experiments call
`chat.*` functions directly, so the backend does not own device, dtype or batching for most
experiments, and nothing can be swapped underneath them.

Two concepts, kept separate:

| concept | question | implementations |
|---|---|---|
| backend | how a model computes | `TorchBackend(device="auto", dtype=...)` replacing `CpuBackend`; later TransformerLens, nnsight, vLLM for generation |
| executor | where a plan runs | local; SSH box (today's `remote.py`); later batch, Modal, a job queue |

Two axes that today's `chat.py` mixes up:
- format: how an example becomes tokens. Token sequences, plain text, or chat (system / user /
  assistant turns, the model's template, an assistant prefix, multi-turn). Chat is a format.
- operation: what is done to the model with those tokens. Inference (score log-probs, read
  activations at positions or spans, generate, loss) or training (fine-tune, LoRA). A chat
  conversation can feed either, which is why they are separate axes.

An experiment is then a composition, registered by stable id (section 14):

```
experiment = data -> format -> intervention(s) -> operation -> readout -> metric

backend/        how a model computes
formats/        tokens, text, chat
operations/     inference: score, read, generate, loss; training: fine-tune, LoRA
interventions/  add vector, ablate direction, zero head, scale neuron (reusable hooks;
                today scattered across experiments: add_hook, ablation_hooks, _neuron_hooks, zero_head)
readouts/       refusal log-odds, balanced A/B probability, judged trait, logit difference
experiments/    compositions of the above
```

Experiments call operations through `ctx`, never a model library directly.

What it buys (the acceptance criteria when this is built):
- New experiments are mostly composition: a Persona-Vectors-style test on a new trait is a new
  data pack plus existing format, intervention, operation and readout pieces, with no new code.
- The backend is swappable: fast generation, TransformerLens or nnsight, and batching across
  claims are backend choices, not rewrites of experiments.
- Training is one more operation, not a special case: V2's fine-tuning experiments (Emergent
  Misalignment) slot in next to inference.

This is the layer for scaling extraction and re-verification in batch, in the spirit of
TransformerLens.

## 17. Expand as dumb machinery (built)

`expand.py` checks for one template by name and calls `templates.contrastive_direction` with
knowledge of that template's parameters (`block`, `layer`, `pack`) and its layer-scaling rule. So
expand works for one experiment family, and every new family would need another branch.

Expand should only be: take the verifications behind a finding, clone them onto another model, run
them, record results and cards. A template's job ends when it has written claims; everything after
works on claims.

What makes that possible: experiments declare what their parameters mean, in the experiment
registry (section 14):

| parameter kind | examples | when moving to another model |
|---|---|---|
| layer index | `block`, `min_layer` | scale to the same relative depth (recorded as supplemented) |
| component index | `neuron`, `head` | not transferable: the closure is refused, with the reason |
| data | `pack`, dataset files | unchanged |
| scalar | `coef`, `threshold` | unchanged |
| artifact input | `direction` | recomputed by the cloned upstream claim, never copied across models |

Expand then:
1. finds the claims about a finding and their dependency closure (direction, steering, controls);
2. clones the closure with `subject.model` replaced, ids re-suffixed, `depends_on` and `inputs`
   rewired to the clones;
3. maps each parameter by its declared kind; a non-transferable parameter refuses the whole closure,
   with the reason (e.g. "neuron indices are model-specific");
4. marks clones `expanded_from` and records each remapping as supplemented;
5. optionally runs `verify` on the clones; cards show them as "beyond the paper".

What it buys: expand works for every experiment family, including V2's agent-written ones, with no
expand-specific code; neuron-steering claims are refused with a reason instead of skipped silently;
templates stop being called by the rest of the system. Depends on the registry (section 14); the
parameter-kind declarations can live beside the experiment functions until then.

## 18. Ingest: source adapters, one extraction pipeline, a stored shape of its own (built: arXiv, web, post; not yet: PDF, GitHub)

Today `ingest.py` is arXiv-specific (API for abstract, authors and date; HTML for the body), and
anything else is "a URL plus pasted text" treated as a post whose whole text is both claims and
evidence. There is no fetcher for blogs (LessWrong, Alignment Forum, transformer-circuits.pub,
personal sites) or X; the evil-Qwen claim was written by hand.

Two layers. What differs per source is how to get the text and where in it claims live versus
evidence:

| source | how to get it | claims | tests |
|---|---|---|---|
| arXiv | API (metadata, abstract) + HTML (body) | abstract | body |
| blog, LessWrong, Alignment Forum, transformer-circuits | fetch HTML to text | TL;DR or summary, else introduction | body |
| X post or thread | pasted text, or the API later | the text | the text, plus its links |
| PDF | text extraction | abstract | body |
| GitHub README | API | summary | README and code |

Everything after that is shared:

```
sources/adapters/  arxiv.py, web.py, x.py, pdf.py, github.py
                   each returns Document {ref, kind, title, authors, date, claims_text, body, links, own_code}
sources/extract.py claims from claims_text, then each claim's tests from body, then checks, then claims
```

Quotes are checked against the field they were taken from (claims_text or body).

Stored shape: `CLAIMS_SCHEMA` and `TESTS_SCHEMA` constrain Claude's output only, but `to_claims`
copies each extracted test verbatim into `stated.tests`, so the prompt's output format quietly
defines part of the stored format and nothing validates it. The stored shape of a source's tests
(and `provenance.checks`) gets its own type in the schema layer (`SourceTest`), checked by
`validate`; the extraction schema maps onto it, and the two evolve separately.
