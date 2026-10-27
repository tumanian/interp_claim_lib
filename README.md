# claimci

A library of interpretability claims in the style of [Gene Ontology](https://geneontology.org/),
where the evidence for a claim is an experiment that reruns.

## Inspiration: Gene Ontology, but the experiments rerun

Gene Ontology curates what is known about genes as annotations ("this gene is involved in DNA
repair"), each with an [evidence code](https://geneontology.org/docs/guide-go-evidence-codes/)
saying what kind of support it has: a mutant phenotype, an author's statement, an automated
inference. Research agents and people query the base instead of rereading the literature.

Interpretability has the same need: many claims about models, of very different strength, spread
across papers and posts. The difference is that its experiments are code. In genomics, "IMP:
inferred from mutant phenotype" records that someone once knocked out a gene in a lab; nobody
reruns it. In interpretability the "mutant" is a forward pass with a hook, so the evidence code
is earned again each time the code runs, on new models, as the code changes.

| | genomics | interpretability |
|---|---|---|
| cost of an experiment | weeks, money, a lab | seconds to hours of compute |
| replication | rare, a project of its own | a rerun; can happen every night |
| the subject | living organisms, variable | a fixed checkpoint; reruns are bitwise identical on one machine |
| generalization | another species is another project | another model is another ledger row |
| development | sampled when someone pays for it | Pythia and OLMo publish every stage of training |
| the expensive part | running the experiment | deciding what the experiment should be |

So the curation model carries over: claims, evidence codes, a base agents can read. And
verification can be continuous. The evidence codes, and where they come from:

| code | meaning here | in GO |
|---|---|---|
| NAS | non-traceable author statement (no section cited) | same |
| TAS | traceable author statement (paper + section or figure) | same |
| IEA | inferred from automated annotation (e.g. an auto-interp label, no experiment) | same idea (electronic annotation) |
| IMP | inferred from mutant phenotype: ablation or patching, behaviour change measured | same idea |
| IOB | inferred from observation: reads activations or attention, no fit, no intervention | ours |
| ICP | inferred from a correlational probe, evaluated held out | ours |
| ISS | inferred from steering: add a direction, the predicted behaviour appears | ours (in GO, ISS means sequence similarity) |
| ICA | inferred from causal abstraction: interchange interventions against a stated high-level model | ours |

Author statements (NAS, TAS) are never verified themselves; a test attached to them earns a
rerunnable code when it reproduces. Claims depend on each other, and the codes constrain how
(no causal claim on a probe alone). The ontology is in
[claimci/schema/evidence.yaml](claimci/schema/evidence.yaml).

## Install and run

```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install -e '.[report,test,ingest]'
claimci serve                                # http://127.0.0.1:8765
```

That is the whole demo: the results below are committed, so reading them needs no keys and no
GPU. Adding a source needs a Claude API key; rerunning claims needs a GPU (see
[Setup in detail](#setup-in-detail)).

## What it does

**1. Given a paper, it extracts the claims, formulates experiments, and runs them on several
models, including ones the paper did not test.**
- `claimci ingest arXiv:2507.21509`: claims come from the abstract, and for each claim the
  paper's own tests (what was done, where, models, data, metric, stated numbers) come from the
  body. Every quote and number is string-checked against the paper. Claude Sonnet 5.5 extracts;
  nothing it says is trusted without the check.
- When the paper's method matches an experiment template (steering directions today: CAA,
  Persona Vectors), matching fills its parameters from the paper's stated tests. Otherwise the
  findings and the paper's own tests are recorded and the finding stays untested until an
  experiment exists for it.
  For Persona Vectors it chose Qwen2.5-7B-Instruct, layer 20, the evil trait; for CAA,
  Llama-2-7b-chat, layer 13, sycophancy and corrigibility. Thresholds are fixed before any run.
- `claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3.5-4B`: the same
  experiments on other models, at the same relative depth. That choice is recorded as ours, and
  a failure there counts as scope, not as evidence against the paper.

**2. It stores the metadata (code, setup, everything) in local YAML**, plus an append-only JSONL
ledger of results, content-addressed artifacts (directions, judged answers) and a data manifest
that says where every file came from (the authors, elsewhere, or made here). Each claim
records which numbers the source stated and which were our choices (`supplemented`).
Trivially a service: the web app (`claimci serve`) already reads it live. The ledger from our
runs is committed (`ledger/`), so cards, `claimci get` and the web app show these results from a
fresh clone.

**3. It re-verifies from the metadata alone, skipping parsing.** `claimci verify` reruns
any claim; results are keyed by the experiment's version, parameters, data, model and
checkpoint, so changing what an experiment computes or the data it reads makes a claim and its
dependents stale, and a rerun of anything unchanged is a cache hit. A refactor that computes the
same thing keeps the version: `claimci validate` fails until each code change is recorded as a
keep or a bump (`claimci versions`). `claimci reverify` runs everything stale, previously
verified claims first, cheapest first. This is where compute goes to keep the knowledge current. Runs go to a GPU box
over SSH when one is configured.

**4. Downstream agents read semi-machine-readable results.** `claimci search`, `get`, `card`
and `explain`, all with `--json`, return verdicts with 95% intervals and margins (a verdict
holds only if the whole interval clears the threshold), the controls they passed,
caveats, whether the threshold was set after seeing results (post hoc), and the evidence type
(e.g. *Causal | Activation steering*). An agent can build on a result (claims depend on other
claims and consume their outputs, e.g. a direction or a head) and add new claims that run
through the same chain.

Not formal verification: a library with evidence codes, controls and provenance, closer to Gene
Ontology than to a proof checker. Next: let the app write the experiments it is missing, safely
([DESIGN.md](DESIGN.md) section 13), so step 1 is not limited to the methods we have templates
for. [SPEC.md](SPEC.md) is the claim format, lifecycle, result record and runner; DESIGN.md is
where this is going and what was built when.

## Results: verified claims and expansions

Rows are findings, columns are models. **(paper)** marks the model the source used; every other
cell is an expansion beyond the source. Values carry 95% intervals (Wilson for rates, a seeded
bootstrap over items for differences); a verdict holds only if the whole interval clears its
threshold.

| finding (source) | gemma-2b-it | Qwen2.5-7B-Instruct | Qwen3.5-4B | Llama-2-7b-chat | pythia-160m |
|---|---|---|---|---|---|
| Refusal is one direction: ablating it stops refusal, adding it induces it ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) | **(paper) ✓ verified**: JailbreakBench refusal 0.91 [0.84, 0.95] → 0.00 [0.00, 0.04] ablated (random direction: 0.91); harmless 0.00 → 1.00 [0.96, 1.00] added; cosine with the authors' direction 0.99999; text loss −0.007 nats | | | | |
| Two neurons make Qwen evil ([@_lyraaaa_ on X](https://x.com/_lyraaaa_/status/2113833330429083662)) | | | **(post) ✓ verified\***: 94% [85, 98] of answers judged misaligned (random neurons: 6% [3, 12]; unsteered: 0% [0, 6]) | | |
| A persona vector steers the trait ([Chen et al. 2025](https://arxiv.org/abs/2507.21509)) | **⊘ could not run**: 2/200 contrast pairs pass the judge, no vector to extract | **(paper) ✓ verified\***: judged evil +64 [59, 69] (random direction: −0.2 [−0.6, 0.05]); 151/200 pairs confirmed | | | |
| Steering vectors shift sycophancy both ways ([Panickssery et al. 2023](https://arxiv.org/abs/2312.06681)) | | **✗ does not transfer**: +0.001 [−0.003, 0.005] / −0.007 [−0.013, −0.003] (threshold ±0.1) | **✗**: +0.011 [0.002, 0.020] / −0.011 | **(paper) not run**: gated | |
| Steering vectors shift corrigibility both ways (same) | | **✗ does not transfer**: +0.024 [0.000, 0.052] / −0.033 [−0.070, −0.004] | **✗ short of the threshold**: +0.12 [0.08, 0.16] up, −0.03 down | **(paper) not run**: gated | |
| Induction heads form early in training ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) | | | | | **✓ verified**: forms between step512 and step1000 (score 0.09 → 0.95); ablation +0.92 nats at the final checkpoint |
| Narrow fine-tuning causes broad misalignment ([Betley et al. 2025](https://arxiv.org/abs/2502.17424)) | | | | | **gap**: needs fine-tuning, which the library does not run yet |

\* loses coherence: 5% of the steered Qwen3.5 answers are coherent (unsteered: 92%), 3% of the
steered Qwen2.5 answers at ×2. The sources claim the trait, not coherence, so coherence is a
caveat on the card rather than part of the verdict. The evil-Qwen threshold was set after seeing
the first results and is marked post hoc. Also ingested, with no test attached yet: Tigges et al.
2023 on linear sentiment directions.

Every cell opens in the web app to its evidence: the quote, the authors' own tests, our
verifications with their thresholds and who set them, controls, and judged samples to read.

## Setup in detail

Python 3.11+.

On a CPU-only machine, install the CPU torch wheel before the package
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`). What else you need
depends on what you run:

| to | you need |
|---|---|
| read results: `validate`, `get`, `card`, `explain`, `search`, `serve` | nothing more |
| `ingest`, `match`, `judge`, or rerun a judged claim (evil-Qwen, Persona Vectors) | `ANTHROPIC_API_KEY` |
| rerun claims on gemma-2b-it or Llama-2 | a Hugging Face token, with each model's license accepted on its Hub page |
| rerun the chat-model claims (up to 7B) | a GPU: local CUDA, or any machine you can SSH into |
| rerun the Pythia induction claims | nothing more (CPU is fine) |

Keys go in `.env` at the repo root (copy [.env.example](.env.example); `.env` is gitignored). The
CLI reads it, and so does a remote box.

**A remote GPU box** (any SSH host; we used an AWS g6.xlarge, one 24 GB L4). Once, on the box:
clone the repo to `~/interp_claim_lib`, create `.venv` and `pip install -e '.[ingest]'` as above,
and put a `.env` there with the same keys (`umask 077` first). Then, locally:

```
echo '{"host": "ubuntu@<ip>", "key": "~/.ssh/<key>.pem", "dir": "~/interp_claim_lib"}' > .claimci-remote.json
```

`verify`, `reverify`, `expand --verify` and the web app's Verify button then push the code and
ledger to the box, run there, and merge the results back. The file is gitignored; an IP that
changes on restart only needs editing here.

## Commands

```
claimci ingest arXiv:2507.21509                   # findings from the abstract, the paper's tests from the body; matched to templates
claimci ingest https://transformer-circuits.pub/...   # a blog or web page (TL;DR / summary section first)
claimci ingest https://x.com/... --text-file post.txt   # a post that cannot be fetched: its URL and its text
claimci match arXiv:2312.06681                    # re-match an ingested source (no re-extraction)
claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3.5-4B
claimci validate                                  # typing rules, provenance, DAG, dataset hashes, versions; no model loads
claimci versions [--keep ID,ID | --bump ID]       # experiments whose code changed: same computation, or new version
claimci plan --claims ID,ID                       # groups, forward passes, ledger hits; no model loads
claimci verify --claims ID,ID                     # run; results to ledger/results.jsonl
claimci reverify [--dry-run]                      # everything stale, previously-verified first, cheapest first
claimci search "refusal direction" [--json]       # find claims
claimci get CLAIM                                 # one result as JSON: verdict, value, margin, controls, caveats
claimci card CLAIM | claimci explain arXiv:ID     # cards, text or --json
claimci judge answers.jsonl                       # the LLM judge as a tool (Claude Haiku 5.5 by default)
claimci serve                                     # the web app
```

Reports for the Pythia run: `claimci matrix --question q.induction_formation --png matrix.png`,
`claimci throughput`, `claimci status`.

## Results: pythia-160m across training

Run on an 8-core Apple Silicon Mac (CPU, torch 2.14.1, transformers 5.21.0),
checkpoints pre-downloaded, with `claimci verify --revisions step0,step1,...,main`.

![matrix](matrix.png)

| claim | evidence | step0 | step1 | step2 | step4 | step8 | step16 | step32 | step64 | step128 | step256 | step512 | step1000 | step2000 | step4000 | step8000 | step16000 | step32000 | step143000 | main |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mech.prev_token_head | IOB | 0.089 ✗ | 0.089 ✗ | 0.089 ✗ | 0.089 ✗ | 0.088 ✗ | 0.094 ✗ | 0.097 ✗ | 0.096 ✗ | 0.097 ✗ | 0.161 ✗ | 0.382 ✗ | 0.930 ✓ | 0.947 ✓ | 0.965 ✓ | 0.970 ✓ | 0.969 ✓ | 0.965 ✓ | 0.949 ✓ | 0.949 ✓ |
| mech.induction_head | IOB | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | 0.948 ✓ | 0.951 ✓ | 0.953 ✓ | 0.951 ✓ | 0.940 ✓ | 0.920 ✓ | 0.878 ✓ | 0.878 ✓ |
| causal.induction_head_ablation | IMP | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | 3.851 ✓ | 1.198 ✓ | 1.400 ✓ | 0.898 ✓ | 0.965 ✓ | 1.071 ✓ | 0.925 ✓ | 0.925 ✓ |

Thresholds: the paper reports no thresholds for Pythia, so every `expect`
here is ours (set when the claims were written) and recorded under
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

## Layout

```
claimci/
  schema/       what a claim is: types, loading, experiment versions as recorded
  checks/       validation as small rule tables: fields, status, evidence, sources, graph, data
  store/        what has been recorded: results ledger (and gating), artifacts, vendored data
  run/          how claims run: plan, runner, verify, backends, executors, formats, operations,
                interventions, readouts
  experiments/  the experiment library, registered by id
  sources/      papers and posts in: arXiv / web / post adapters, extraction, matching, templates,
                expansion to other models
  read/         reading results: report cards, matrix, objects, agent API, judge tool
  app/          the local web app
  cli.py
```

Imports point one way (schema <- store <- run <- experiments; sources and read on top; app and
cli above everything), checked by tests/test_layout.py. Claim keys do not depend on this layout.

## Tests

```
pytest
```

107 tests. Tests never download and never call an API (the judge is stubbed). What needs real
weights skips until they are in the HF cache: `EleutherAI/pythia-70m` (`main`, `step0`) for the
induction checks, `Qwen/Qwen2.5-0.5B-Instruct` for end-to-end runs of the steering experiments
through `verify` (CAA and Persona Vectors: direction, steering, random control, artifacts, ledger
cache, hook cleanup). Everything else runs on a random-init model with Pythia's architecture,
built locally, or on no model at all (schema, validation, ingest checks, templates, matching,
expansion, reports, search, re-verification order).

## License

MIT for the code ([LICENSE](LICENSE)). Vendored datasets keep their own licenses, recorded with
their sources in [data/MANIFEST.yaml](data/MANIFEST.yaml).
