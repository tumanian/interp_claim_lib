# claimci claim spec (v0.2)

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

```
proposed ──verify──> verified ──(code or data changed)──> stale ──verify──> verified
    │                    │
    │ (rerun off)        └──> refuted (NOT qualifier recorded, same evidence standard)
    └──> unreproduced
```

- `TAS`/`NAS` claims enter as `proposed` and stay there until a rerunnable
  experiment is attached and reproduces the stated value within tolerance.
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
  entrypoint: claimci.experiments.induction:ablate_head_loss_delta
  version: git:a3f9c2e
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
claim_key: sha(code.version + params + data.hash)
value: 0.57
holds: true
bindings: {layer: 4, head: 11}
reproduction: {target: 0.4, within_tolerance: true}   # only for promotion runs
cost: {forward_passes: 1, seconds: 3.2}
run: {runner: claimci 0.1, host: ..., at: 2026-10-26T21:00Z}
```

## Typing rules enforced by the runner

1. `NAS`, `TAS`, `IEA` claims cannot appear in `depends_on`.
2. A claim with `evidence` in {IMP, ISS, ICA} must have `code.params` or an
   entrypoint tagged as an intervention; `ICP`/`IOB` claims cannot be relabeled
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
