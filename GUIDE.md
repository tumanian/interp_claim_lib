# claimci guide

## Why Gene Ontology, and what changes

Gene Ontology curates what is known about genes as annotations, each with an
[evidence code](https://geneontology.org/docs/guide-go-evidence-codes/) saying what kind of support
it has. In genomics, "IMP: inferred from mutant phenotype" records that someone once knocked out a
gene in a lab; nobody reruns it. In interpretability the "mutant" is a forward pass with a hook,
so the evidence code is earned again each time the code runs.

| | genomics | interpretability |
|---|---|---|
| cost of an experiment | weeks, money, a lab | seconds to hours of compute |
| replication | rare, a project of its own | a rerun; can happen every night |
| the subject | living organisms, variable | a fixed checkpoint; reruns are bitwise identical on one machine |
| generalization | another species is another project | another model is another ledger row |
| development | sampled when someone pays for it | Pythia and OLMo publish every stage of training |
| the expensive part | running the experiment | deciding what the experiment should be |

So the curation model carries over (claims, evidence codes, a base agents can read) and
verification can be continuous. Objects are anchors, not claims: the web app gathers everything
checked about one component, across sources, in one place.

Claims are about three kinds of thing, as in biology:

| layer | in interpretability | examples | in biology |
|---|---|---|---|
| object | a component with an identity in one model | neuron L14N8949 of Qwen3.5-4B; head L4H6 of pythia-160m; the refusal direction of gemma-2b-it | a gene or variant ([Ensembl](https://www.ensembl.org/), [dbSNP](https://www.ncbi.nlm.nih.gov/snp/)) |
| annotation | a claim that one object does something | "L4H6 is an induction head"; "these two neurons push Qwen toward evil" | a GO or [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) annotation: function, with an evidence code |
| mechanism | a claim about how a computation works, across objects | "refusal is mediated by a single direction"; "induction heads form in a phase change" | a pathway ([Reactome](https://reactome.org/)) |

## Evidence codes

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

## Results: verified claims and expansions

One table per study. **original** is the model the source used (for linear sentiment, our first
run, on one of the paper's model families); every other row is an expansion. Values carry 95%
intervals (Wilson for rates, a seeded bootstrap over items for differences). On the original model
a test **holds** (✓) only if its whole interval clears the threshold. On an expansion, a test with a
random control is judged by effect instead, since the thresholds were set for the source's model:
it **transfers** when its interval clears the control's in the predicted direction, **weaker** when
that effect is under half the original's, and **doesn't transfer** otherwise; tests without a
control keep the threshold. ⊘ could not run (see the card for why).

**[Persona Vectors (Chen et al. 2025)](https://arxiv.org/abs/2507.21509)**

| model | | steering raises the trait | monitoring (AUROC) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | **original** | ✓ +64 [59, 69] | ✓ 1.00 [1.00, 1.00] |
| Qwen3.5-4B | expanded | ✓ weaker +29 [23, 35] | ✓ 1.00 [1.00, 1.00] |
| gemma-2-2b-it | expanded | ✓ transfers +37 [32, 42] | ✓ 1.00 [1.00, 1.00] |
| gemma-2b-it | expanded | ⊘ could not run |  |

**[Refusal direction (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)**

| model | | ablating removes refusal | adding induces refusal | capabilities kept (Δ loss) |
|---|---|---|---|---|
| gemma-2b-it | **original** | ✓ 0.00 [0.00, 0.04] | ✓ 1.00 [0.96, 1.00] | ✓ -0.01 |
| Qwen2.5-7B-Instruct | expanded | ✓ transfers 0.00 [0.00, 0.04] | ✓ weaker 0.46 [0.37, 0.56] | ✗ 0.31 |
| Qwen3.5-4B | expanded | ✗ doesn't transfer 0.85 [0.77, 0.91] | ✓ weaker 0.24 [0.17, 0.33] | ✓ -0.01 |
| gemma-2-2b-it | expanded | ✓ transfers 0.16 [0.10, 0.24] | ✓ transfers 0.99 [0.95, 1.00] | ✓ -0.00 |

**[Linear sentiment (Tigges et al. 2023)](https://arxiv.org/abs/2310.15154)**

| model | | ablation removes SST accuracy | direction reads SST | commas only |
|---|---|---|---|---|
| pythia-1.4b | **original** | ✓ 1.00 [1.00, 1.00] | ✓ 0.83 [0.77, 0.87] | ✗ 0.07 [0.00, 0.15] |
| pythia-2.8b | expanded | ✓ transfers 1.00 [1.00, 1.00] | ✓ 0.77 [0.71, 0.82] | ✗ 0.00 [0.00, 0.00] |
| pythia-410m | expanded | ✓ transfers 0.87 [0.64, 1.00] | ✗ 0.76 [0.69, 0.81] | ✗ 0.00 [0.00, 0.00] |
| gpt2 | expanded | ⊘ could not run | ✗ 0.52 [0.44, 0.58] | ⊘ could not run |

**[CAA (Panickssery et al. 2023)](https://arxiv.org/abs/2312.06681)**

| model | | corrigibility up | sycophancy up |
|---|---|---|---|
| Llama-2-7b-chat-hf | **original** | not run (gated) | not run (gated) |
| Qwen2.5-7B-Instruct | expanded | ✗ doesn't transfer 0.02 [-0.00, 0.05] | ✗ doesn't transfer 0.00 [-0.00, 0.00] |
| Qwen3.5-4B | expanded | ✓ transfers 0.12 [0.08, 0.16] | ✗ doesn't transfer 0.01 [0.00, 0.02] |
| gemma-2-2b-it | expanded | ✓ transfers 0.18 [0.08, 0.29] | ✗ doesn't transfer 0.00 [-0.00, 0.01] |
| gemma-2b-it | expanded | ✗ doesn't transfer 0.00 [-0.01, 0.01] | ✗ doesn't transfer 0.00 [-0.00, 0.00] |

Also verified, on the source's own model: two neurons make Qwen3.5-4B evil
([@_lyraaaa_ on X](https://x.com/_lyraaaa_/status/2113833330429083662)): 94% [85, 98] of answers
judged misaligned, against 6% [3, 12] for random neurons (the steered model loses coherence);
induction heads form between step512 and step1000 of pythia-160m
([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)). Every row opens in the web app to its
evidence: quotes, the authors' tests, our thresholds and who set them, controls, judged samples.

## What it does, in detail

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

## Setup

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
claimci export-site --out site [--base /path/]    # the web app as a read-only static site (Vercel, any static host)
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
