# claimci

**A library of interpretability claims where the evidence is an experiment that reruns.**
Findings come from papers and posts. Each gets tests with pre-set thresholds, run on the
source's own model and then on others. Results live in an append-only ledger. When a model,
dataset or experiment changes, the affected claims go stale and rerun.

Read-only demo: [tumanian.dev/interp_lib_demo](https://tumanian.dev/interp_lib_demo/)

Inspired by Goodfire's [Accelerating Interpretability](https://www.goodfire.com/blog/accelerating-interpretability)
essay (footnote 2: a database of typed causal claims backed by experimental code) and by the
[Gene Ontology](https://geneontology.org/). Where GO annotates a gene product with a term and an
evidence code, claimci annotates a model component (a head, a direction, a neuron) with a
statement, a metric, a threshold and an evidence code. The difference: in interpretability the
experiment is a forward pass, so the evidence code is earned every time the code runs. Any claim
in a paper, or a tweet, can be replicated and checked for transfer across models.

## What it found

Six sources, 79 claims, run on five model families. **(paper)** marks the source's own model;
other cells are expansions. Values carry 95% intervals; a verdict holds only if the whole
interval clears the threshold.

| finding (source) | gemma-2b-it | Qwen2.5-7B-Instruct | Qwen3.5-4B | pythia |
|---|---|---|---|---|
| Refusal is one direction ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) | **(paper) ✓** refusal 0.91 → 0.00 ablated, 0.00 → 1.00 added; random direction 0.91 | | | |
| Two neurons make Qwen evil ([@_lyraaaa_](https://x.com/_lyraaaa_/status/2113833330429083662)) | | | **(post) ✓** 94% misaligned; random neurons 6% | |
| Persona vectors steer and monitor a trait ([Chen et al. 2025](https://arxiv.org/abs/2507.21509)) | ⊘ could not run | **(paper) ✓** judged evil +64; monitor AUROC 1.00 | | |
| Steering vectors shift sycophancy ([Panickssery et al. 2023](https://arxiv.org/abs/2312.06681)) | | **✗** +0.001 (threshold 0.1) | **✗** +0.011 | |
| Sentiment is a linear, causal direction ([Tigges et al. 2023](https://arxiv.org/abs/2310.15154)) | | | | **(paper family, 1.4b)** ✓ ablation removes 100% of SST accuracy; ✗ 2 of 4 sub-findings |
| Induction heads form early in training ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) | | | | **(160m) ✓** forms between step512 and step1000 |

![Induction head claims across 18 Pythia-160M training checkpoints](matrix.png)

Every cell opens in the web app to its evidence: the quote, the authors' tests, our thresholds and
who set them, controls, judged samples. Full table with intervals in [GUIDE.md](GUIDE.md).

A "✗" means the test did not clear its threshold on that model. It can mean the paper's claim does
not transfer, or that our test is a poor fit; the claim card says which the evidence supports.

## What a claim looks like

```yaml
id: refusal.gemma2b.direction
kind: test
evidence: IOB
statement: Difference in means at layer 10, position -2, on the paper's own training
  samples, reproduces the authors' published gemma-2b-it direction (cosine >= 0.99).
about: src.arditi2024.refusal_one_direction      # the finding this tests
subject: {model: {hf: google/gemma-2b-it, revision: main}}
code:   {experiment: refusal.direction, params: {layer: 10, pos: -2}}
data:   {dataset: {gen: vendored, params: {...}}, hash: 6c93d3eb...}
metric: cosine_with_published
expect: {op: '>=', value: 0.99}
exports: [direction, layer]                      # downstream claims bind to these
provenance:
  supplemented:                                  # every number that is ours, not the paper's
    expect: {by: agent:claude-code, note: "same method and data should give nearly the same vector"}
```

The claim key is a hash of the experiment version, params, data hash, model and checkpoint.
Change any of them and the claim is stale. Thresholds are set before a run; one set after seeing
a value is flagged `post_hoc`. Nothing in the ledger is written by hand.

## Evidence codes

Borrowed from GO. Author statements are never verified themselves; a test attached to them earns
a code when it runs.

| code | meaning | reruns | can be depended on |
|---|---|---|---|
| NAS / TAS | author statement (non-traceable / traceable to a section) | no | no |
| IOB | observation: reads activations or attention, no fit, no intervention | yes | yes |
| ICP | correlational probe, held-out eval | yes | not by causal claims |
| IMP | ablation or patching, behavior change measured | yes | yes |
| ISS | steering: add a direction, predicted behavior appears | yes | yes |
| ICA | causal abstraction: interchange intervention vs a stated high-level model | yes | yes |

Controls (`control_for`) are tests that must *not* reproduce the effect, such as a random direction.

## Reading results (no keys, no GPU)

```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install -e '.[report,test,ingest]'

claimci serve                                   # web app at http://127.0.0.1:8765
claimci search "steering" --json                # find claims
claimci get refusal.gemma2b.ablation_jbb        # verdict, value, interval, threshold and who set it, controls
claimci explain arXiv:2406.11717                # every finding of a source with its tests
claimci status                                  # counts by status and evidence; what is stale and why
```

## Adding and verifying claims

Ingest a paper (needs `ANTHROPIC_API_KEY` in `.env`): findings from the abstract, every quote
checked against the body.

```
claimci ingest arXiv:2310.15154
```

Verify (needs a GPU box you can SSH into; we used an AWS g6.xlarge):

```
echo '{"host": "ubuntu@<ip>", "key": "~/.ssh/<key>.pem", "dir": "~/interp_claim_lib"}' > .claimci-remote.json
claimci verify --claims sent.pythia14b.ablate   # runs on the box, results merge into ledger/
claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct --verify
claimci reverify --dry-run                      # what is stale, in the order it would rerun
```

Agents: run `claude` in the repo. [CLAUDE.md](CLAUDE.md) gives it the vocabulary, commands and
rules, so it can answer "which steering claims hold on Qwen, and with what margin?" or add a
verification on a new model and run it.

## What this is not

Not formal verification. A claim holds to a threshold, on a dataset, within an interval. The
model is GO-style curation with continuous rerun, not Lean. Several independent tests per
finding is how the verdict becomes about the paper rather than about one test.

## Next

Batch and scale `reverify`: group claims by model and dataset, share forward passes, run the
backlog nightly, verified claims first, cheapest first.

More: [GUIDE.md](GUIDE.md) (setup, commands, results in detail), [SPEC.md](SPEC.md) (formats),
[DESIGN.md](DESIGN.md) (where this is going). MIT license.
