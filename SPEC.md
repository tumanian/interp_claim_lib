# claimci spec

A claim is experiment code applied to versioned data, with typed evidence and
provenance. Humans and agents write claims. Only the runner writes results.

Modeled on Gene Ontology annotations (GAF) and evidence codes (ECO).

## Evidence codes

| Code | Name | Rerunnable | Can be a dependency |
|---|---|---|---|
| NAS | Non-traceable author statement (claim from a paper, no section cited) | no | no |
| TAS | Traceable author statement (paper + section/figure cited) | no | no |
| IEA | Inferred from automated annotation (auto-interp label, no experiment) | yes | no |
| IOB | Inferred from observation (reads activations/attention, no fit, no intervention) | yes | yes |
| ICP | Inferred from correlational probe (fit on activations, held-out eval) | yes | yes, not for causal claims |
| IMP | Inferred from mutant phenotype (ablation / patching, behavior change measured) | yes | yes |
| ISS | Inferred from steering (add direction/feature, predicted behavior appears) | yes | yes |
| ICA | Inferred from causal abstraction (interchange intervention vs stated high-level model) | yes | yes |

Codes live in `evidence.yaml` as a small ontology (id, parent, definition),
not an enum in code.

## Claim lifecycle

A claim file stores only decisions (claimci/schema/lifecycle.py):

```
proposed ──run──> verified          a verification run reproduced the source (verify --promote)
proposed ──run──> unreproduced      it ran and did not
unreproduced ──run──> verified      a later run did
any ──person──> refuted             the opposite finding, recorded with qualifier NOT
refuted ──person──> proposed        withdrawn
```

What a reader sees beyond that is derived from the ledger on read and never written: measured
(a value, no threshold), stale (the claim's current key has no row), not holding, gated (a
parent does not hold).

- `TAS`/`NAS` claims enter as `proposed` and stay there until a rerunnable
  experiment is attached and reproduces the stated value within tolerance.
- When the source states no value, the threshold in `expect` is ours, not the
  source's, and must be recorded under `provenance.supplemented.expect` (who set
  it, why); `validate` rejects a proposed rerunnable claim without one. Such a
  claim is promoted when `expect` holds at its subject revision, and both the
  ledger row and the claim file record `basis: supplemented_expect` (vs
  `stated_value` when the source's number was reproduced within tolerance).
- Promotion: when a `proposed` claim reproduces, `evidence` becomes the code the
  experiment earned (IMP, ICP, ...) and `reproduced_from` records the source.
- Refutation is a result, not a deletion. `qualifier: NOT` with full provenance.

## Claim file

```yaml
id: causal.induction_head_ablation
question: q.induction_formation
qualifier: ""                        # "" or NOT
evidence: IMP                        # current earned code (TAS/NAS until reproduced)
status: verified                     # proposed | verified | stale | refuted | unreproduced
statement: >
  Zero-ablating the head bound by mech.induction_head raises mean loss on the
  second copy of repeated random tokens by >= 0.3 nats.

subject:                             # the model component the claim is about
  model: {hf: EleutherAI/pythia-160m, revision: step10000}
  component: {layer: $layer, head: $head}   # bound from inputs

code:
  experiment: induction.ablate_head   # a registered id (@experiment), never a Python path
  # version: git:a3f9c2e              # only for external code pinned by commit; library code is
                                      # versioned in claimci/experiments/versions.yaml
  params: {ablation: zero}
  env: lock/induction.txt

data:
  inherit: mech.induction_head       # same model + dataset => shares the forward pass
  # or explicit:
  # dataset: {gen: repeated_random_tokens, params: {n: 48, half: 20, seed: 0}}
  # hash: 7d1e...

metric: loss_delta_nats
expect: {op: ">=", value: 0.3}
tolerance: 0.05                      # for reproduction against a published value

depends_on: [mech.induction_head]
inputs: {layer: mech.induction_head.layer, head: mech.induction_head.head}
exports: []

provenance:
  source:                            # where the claim came from
    kind: paper                      # paper | agent | human
    ref: arXiv:2209.11895
    locator: "Section 3, Fig. 4"     # required for TAS, absent for NAS
    stated_value: 0.4                # the number the source reports, if any
    stated_evidence: IMP             # the evidence the source used, as the source describes it
    code: {repo: github.com/..., commit: ...}   # if the source ships code
  reproduced_from: arXiv:2209.11895  # set by runner on successful promotion
  assigned_by: agent:ingest-v0.1     # who proposed it
  date: 2026-10-26
```

## Result record (runner-only, appended to ledger)

```yaml
claim: causal.induction_head_ablation
claim_key: 5806edc89a9b360c        # sha of experiment id + version, params, inputs, data, model, checkpoint, upstream keys
status: ok                         # ok | error | blocked | contaminated
value: 0.57
interval: [0.51, 0.63]             # 95%, when the experiment reports one (Wilson or bootstrap)
bindings: {layer: 4, head: 11}     # exports and details; judged answers as an artifact reference
model: EleutherAI/pythia-160m
revision: main
code_version: v1                   # the experiment's version (claimci/experiments/versions.yaml)
data_hash: 7d1e...
lineage: {claim_id: [code_version+params, data ident], ...}   # self and ancestors
traceback: artifact:...            # errors only
cost: {forward_passes: 1, seconds: 3.2, backend: torch}
run: {runner: claimci 0.1.0, host: ..., run_id: ..., at: 2026-10-27T21:00Z}
```

Values are always recorded; verdicts are not. `holds`, the margin and promotion's reproduction are
computed on read from the claim's current `expect`, so editing a threshold re-judges every stored
value without a rerun. With an interval, the whole interval must clear the threshold.

## Typing rules enforced by the runner

1. `NAS`, `TAS`, `IEA` claims cannot appear in `depends_on`.
2. A claim with `evidence` in {IMP, ISS, ICA} must have `code.params` or an
   experiment tagged as an intervention; `ICP`/`IOB` claims cannot be relabeled
   causal by editing the field.
3. `ICP` claims cannot be dependencies of causal claims.
4. `exports` may only be consumed on the same data version unless the consumer
   declares `cross_version: true`.
5. `claim_key` ignores `statement`, `provenance`, and `expect`; those change
   interpretation, not computation. Changing `expect` re-evaluates `holds`
   from the stored value without rerunning.

## Ingestion capability (literature → claims)

An ingest agent reads a paper and emits claim files with:
- `evidence: TAS` (or `NAS` if no locator), `status: proposed`
- `provenance.source` filled: ref, locator, stated_value, stated_evidence, code if any
- `subject`, `statement`, `metric`, `expect` as the paper states them

The runner then:
1. Attaches an experiment (from the paper's code via an adapter, or from the
   local experiment library when one matches the stated_evidence and metric).
2. Runs on the paper's model and data. Within `tolerance` of `stated_value`:
   promote. Otherwise: `status: unreproduced`, keep the record.
3. Queues promoted claims for generalization across checkpoints and models.

Adapters (TransformerLens, nnsight, plain HF) supply the model handle and a
shared activation cache so external code runs unchanged.

## Questions

```yaml
id: q.induction_formation
text: Does this model form induction heads, and when in training?
status: open                         # open | answered | superseded
claims: [mech.prev_token_head, mech.induction_head, causal.induction_head_ablation]
parent: q.in_context_learning
```

## Non-goals for v0

Distributed execution, disk activation cache, the arXiv scraper itself,
GO-CAM-style circuit graphs linking claims into mechanisms.

## Runner

```
load ──> validate ──> plan ──> execute ──> record ──> report
```

- **load**: claims, questions, evidence codes; `data.inherit` chains resolved; each claim's code
  version read from claimci/experiments/versions.yaml (or a pinned `git:<sha>`).
- **validate** (before any model loads): the typing rules above, a DAG of `depends_on`, every
  `inputs` reference naming an `exports` of a dependency, declared datasets regenerating to their
  declared hash, and experiment versions up to date with the code (`claimci versions`).
- **plan** (`claimci plan`, printable without running): skip claims whose key already has a
  ledger row (unless `--force`); group the rest by (model, checkpoint, dataset), one shared pass
  per group; order groups to load each model once; sort each group by `depends_on`.
- **execute**: on a backend (how a model computes: `torch`, on CPU or GPU) through an executor
  (where the plan runs: here, or a GPU box over SSH via `.claimci-remote.json`). Observational
  claims read the shared pass; interventional ones rerun the model with hooks registered in a
  context manager and removed on exit. After each intervention a cheap canary probe must match
  the shared pass, else the claim and the rest of its group are `contaminated` and the model is
  reloaded. Per-claim timeout; a claim that raises records `error` with its traceback and does
  not block siblings; dependents of a failed claim are `blocked`.
- **record**: append-only JSONL, one row per (claim key, run); content-addressed artifacts for
  directions and judged answers. Promotion (`verify --promote`) is the only write to a claim file.
- **report**: `claimci matrix`, `throughput` (shared vs naive passes), `status`, `card`, `get`.

What reruns:

| Change | Reruns |
|---|---|
| An experiment's computation (version bumped) | every claim on it, and dependents |
| A refactor (version kept) | nothing |
| A claim's params or inputs | that claim, and dependents |
| A dataset or its generator | every claim on that data |
| A new checkpoint | every claim, that checkpoint only |
| `expect`, `statement`, `provenance` | nothing; verdicts are derived on read |
