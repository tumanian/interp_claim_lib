# claimci

CI for mechanistic interpretability claims.

A claim is a statement about a model component, paired with the experiment code
that tests it and the versioned data it runs on. Verifying a claim means
rerunning that code; the runner appends the result to a ledger, keyed by a hash
of code, params, data, model and checkpoint. Editing code or data makes the
claim and its dependents stale, while editing the threshold (`expect`) only
re-evaluates stored values.

Specs: [SPEC.md](SPEC.md) (claim format, evidence codes, lifecycle) and
[RUNNER.md](RUNNER.md) (pipeline, planner, backend, invalidation).
[PENDING.md](PENDING.md) lists the checks that still need a machine with
Hugging Face access.

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

The ontology lives in [claimci/evidence.yaml](claimci/evidence.yaml).

## Install

Python 3.11+. On CPU-only machines, install the CPU torch wheel first so pip
doesn't pull the CUDA build:

```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, CPU-only
pip install -e '.[report,test]'
claimci validate && claimci plan --revisions main
```

`validate` and `plan` never load a model.

## Commands

```
claimci validate                                  # typing rules, DAG, dataset hashes; no model loads
claimci plan --revisions main,step1000            # groups, forward passes, ledger hits; no model loads
claimci verify --revisions main,step1000          # run, append results to ledger/results.jsonl
```

Reports, read from the ledger:

```
claimci matrix --question q.induction_formation --png matrix.png
claimci throughput
claimci status
```

`verify` options: `--mode naive` (one clean pass per claim, the throughput
baseline), `--force`, `--claims`, `--offline`, `--claim-timeout`,
`--mem-budget`, `--promote`.

## Results: pythia-160m across training

Run on an 8-core Apple Silicon Mac (CPU, torch 2.14.1, transformers 5.21.0),
checkpoints pre-downloaded. `scripts/step8.sh` reproduces the eight-checkpoint
run; the early checkpoints (step1..step512) and `main` were added with
`claimci verify --revisions ...`.

![matrix](matrix.png)

| claim | evidence | step0 | step1 | step2 | step4 | step8 | step16 | step32 | step64 | step128 | step256 | step512 | step1000 | step2000 | step4000 | step8000 | step16000 | step32000 | step143000 | main |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mech.prev_token_head | IOB | 0.089 ✗ | 0.089 ✗ | 0.089 ✗ | 0.089 ✗ | 0.088 ✗ | 0.094 ✗ | 0.097 ✗ | 0.096 ✗ | 0.097 ✗ | 0.161 ✗ | 0.382 ✗ | 0.930 ✓ | 0.947 ✓ | 0.965 ✓ | 0.970 ✓ | 0.969 ✓ | 0.965 ✓ | 0.949 ✓ | 0.949 ✓ |
| mech.induction_head | IOB | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | 0.948 ✓ | 0.951 ✓ | 0.953 ✓ | 0.951 ✓ | 0.940 ✓ | 0.920 ✓ | 0.878 ✓ | 0.878 ✓ |
| causal.induction_head_ablation | IMP | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | 3.851 ✓ | 1.198 ✓ | 1.400 ✓ | 0.898 ✓ | 0.965 ✓ | 1.071 ✓ | 0.925 ✓ | 0.925 ✓ |

Thresholds: the paper reports no thresholds for Pythia, so every `expect`
here is ours (set in PROMPT.md) and recorded under
`provenance.supplemented.expect` in each claim file. All three claims were
promoted to `verified` at their subject revision (`main`) with
`basis: supplemented_expect`, meaning they hold under our thresholds, not that
a number from the paper was reproduced.

Onset: the induction head forms between step512 and step1000 (about 1-2B
tokens). With the dependency gate removed in a throwaway copy, so the
blocked cells have values, the induction score is 0.04-0.09 from step0 through
step512 and 0.948 at step1000; the ablation loss delta goes from 0.007 to
3.85 nats. The previous-token head rises a little earlier (0.161 at step256,
0.382 at step512). Bound heads: previous-token L3H2 at every trained
checkpoint; induction L4H10 at step1000, L4H6 from step2000 on.

Throughput (eight-checkpoint run):

| mode | claims verified | forward passes | seconds | claims/minute |
|---|---|---|---|---|
| shared | 22 | 15 | 24.2 | 54.6 |
| naive | 22 | 29 | 33.6 | 39.2 |

Forward passes naive/shared: 1.93x.

## Tests

```
pytest
```

Tests never download. Tests that need trained weights skip until
`EleutherAI/pythia-70m` (`main`, `step0`) is in the HF cache. The rest of the
suite runs on a random-init model with the same architecture, built locally.
