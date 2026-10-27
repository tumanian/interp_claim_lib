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
checkpoints pre-downloaded. Reproduce with `scripts/step8.sh`.

```
claimci verify --revisions step0,step1000,step2000,step4000,step8000,step16000,step32000,step143000
claimci matrix --question q.induction_formation --png matrix.png
claimci throughput
```

![matrix](matrix.png)

| claim | evidence | step0 | step1000 | step2000 | step4000 | step8000 | step16000 | step32000 | step143000 |
|---|---|---|---|---|---|---|---|---|---|
| mech.prev_token_head | IOB | 0.089 ✗ | 0.930 ✓ | 0.947 ✓ | 0.965 ✓ | 0.970 ✓ | 0.969 ✓ | 0.965 ✓ | 0.949 ✓ |
| mech.induction_head | IOB | blocked | 0.948 ✓ | 0.951 ✓ | 0.953 ✓ | 0.951 ✓ | 0.940 ✓ | 0.920 ✓ | 0.878 ✓ |
| causal.induction_head_ablation | IMP | blocked | 3.851 ✓ | 1.198 ✓ | 1.400 ✓ | 0.898 ✓ | 0.965 ✓ | 1.071 ✓ | 0.925 ✓ |

Bound heads: previous-token L3H2 at every trained checkpoint; induction L4H10
at step1000, L4H6 from step2000 on. The induction head already holds at
step1000, the first trained checkpoint here, so this grid places the onset
before step1000 but does not resolve it further.

Throughput:

| mode | claims verified | forward passes | seconds | claims/minute |
|---|---|---|---|---|
| shared | 22 | 15 | 24.2 | 54.6 |
| naive | 22 | 29 | 33.6 | 39.2 |

Forward passes naive/shared: 1.93x. (24 claim-checkpoint pairs were run; the
2 blocked at step0 are not counted as verified, and step0 skips its
intervention pass.)

## Tests

```
pytest
```

Tests never download. Tests that need trained weights skip until
`EleutherAI/pythia-70m` (`main`, `step0`) is in the HF cache. The rest of the
suite runs on a random-init model with the same architecture, built locally.
