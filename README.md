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

Not yet run: this was built in an environment without access to Hugging Face.
`scripts/step8.sh` produces the matrix and the throughput table; paste them here.

```
claimci verify --revisions step0,step1000,step2000,step4000,step8000,step16000,step32000,step143000
claimci matrix --question q.induction_formation --png matrix.png
claimci throughput
```

Matrix: _pending_

Throughput: _pending_

## Tests

```
pytest
```

Tests never download. Tests that need trained weights skip until
`EleutherAI/pythia-70m` (`main`, `step0`) is in the HF cache. The rest of the
suite runs on a random-init model with the same architecture, built locally.
