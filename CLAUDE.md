# claimci, for Claude

A library of interpretability claims where the evidence is an experiment that reruns. Findings
come from papers and posts; verifications are experiments with thresholds; results are rows in an
append-only ledger. Read results through the CLI, never by parsing files by hand.

## Vocabulary

- **finding** (`kind: finding`, ids `src.*`): what a source asserts, quoted from its abstract.
  Never verified itself; its `level` is `annotation` (one object does something) or `mechanism`.
- **verification** (`kind: test`): an experiment with a threshold (`expect`), `about:` a finding.
- **readout** (`kind: measurement`): an experiment with a value and no threshold.
- **control** (`control_for:`): a test that must *not* reproduce the effect (e.g. a random direction).
- **original / expanded**: run on the source's own model, or on another one (`provenance.expanded_from`).
- **evidence codes**: TAS/NAS (author statements), IOB (observation), ICP (probe), IMP (ablation),
  ISS (steering), ICA (causal abstraction). claimci/schema/evidence.yaml.
- Verdicts are derived on read, never stored: a test **holds** only if its value's whole 95%
  interval clears the threshold. On an **expanded** model, a test with a random control is judged
  by effect instead: **transfers** (its interval clears the control's in the predicted direction),
  **transfers, weaker** (under half the effect on the source's model), or **doesn't transfer**. A test is **gated** if a parent does not hold; **stale** if its
  key (experiment version, params, data, model, checkpoint, parents) has no ledger row.

## Reading results (no keys, no GPU)

```
claimci search "<words>" --json [--kind finding|test|measurement] [--model <substring>] [--evidence ISS] [--limit 50]
claimci get <claim-id>                 # verdict, value, interval, margin, threshold and who set it, controls, caveats, stale, run_on
claimci card <claim-id> --json         # one claim with its evidence and results
claimci explain <source> --json        # every finding of a source with its tests: arXiv:2310.15154, a URL, or a short name (tigges2023)
claimci status                         # counts by status and evidence; what is stale and why
claimci reverify --dry-run             # what would rerun, in order
```

`search` matches words in ids, statements, quotes, sources, models, experiments and evidence
(code and label: "steering" finds ISS tests), returns at most `--limit` results (10 by default),
and with an empty query lists everything that passes the filters. Every verification in `search`,
`get` and `card` carries `run_on`: `original` (the source's model) or `expanded` (with
`expanded_from`). Short source names are the part after `src.` in finding ids.

When you report a result, give the verdict with its value and interval, say whether it ran on the
source's model or an expansion, and pass on caveats and post hoc flags. "fails" can mean the
paper is wrong or our test is a poor fit; say which the evidence supports.

## Adding claims

1. Copy a similar claim from `claims/` (e.g. `claims/sent.pythia14b.ablate.yaml`). Required:
   `id`, `kind`, `question`, `evidence`, `status: proposed`, `statement`, `about`, `subject.model`,
   `code.experiment` + `params`, `data`, `metric`, `expect` for a test.
2. Every number that is our choice, not the source's, goes under `provenance.supplemented.<field>`
   with `by` and a `note`. Set thresholds **before** running; if a threshold is set after seeing a
   value, add `provenance.post_hoc: true`. Never move a threshold to make a claim pass.
3. Add the claim id to its `questions/q.*.yaml`, then `claimci validate` (must print `ok`).
4. `claimci verify --claims <id>`: runs locally, or on the GPU box if `.claimci-remote.json`
   exists. Results go to `ledger/`; never edit the ledger by hand.
5. New findings come from `claimci ingest <arXiv id | URL>` (needs `ANTHROPIC_API_KEY` in `.env`),
   which extracts from the abstract and checks every quote. Do not hand-write findings from memory.

To try a finding on another model: `claimci expand <source-ref> --models <hf id>[,<hf id>] [--verify]`.

## Changing experiment code

Experiments live in `claimci/experiments/` as readable recipes: `fn(ctx, **params) -> Outcome`,
parameter kinds by type hint (`block: Layer`, `pack: Data`, `direction: Input`). After editing
anything an experiment runs, `claimci validate` fails until you record a decision:

- `claimci versions --keep <id>`: same computation (a refactor). Show it: run old and new code
  on the same inputs (a tiny random model is enough) and compare values.
- `claimci versions --bump <id>`: different computation; its claims go stale and must rerun.

## Working in this repo

- Tests: `pytest` (107; no downloads, no API calls). Model runs bigger than Pythia-70M belong on
  the GPU box, not a laptop.
- Imports point one way (tests/test_layout.py): schema <- store <- run <- experiments; checks,
  sources and read on top; app and cli above everything.
- Work on a branch and open a PR into `main`. Keys live in `.env` (gitignored); never print them.
- More: GUIDE.md (setup, commands, results), SPEC.md (formats), DESIGN.md (plans, what was built).
