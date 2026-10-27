# claimci

A library of interpretability claims in the style of [Gene Ontology](https://geneontology.org/),
where the evidence for a claim is an experiment that reruns.

Gene Ontology curates what is known about genes as annotations, each with an
[evidence code](https://geneontology.org/docs/guide-go-evidence-codes/) saying what supports it.
Interpretability needs the same for claims about models, with one difference: its experiments are
code, so the evidence is earned again each time the code runs, on new models, as the code changes.
Claims are about three kinds of thing, as in biology:

| layer | in interpretability | examples | in biology |
|---|---|---|---|
| object | a component with an identity in one model | neuron L14N8949 of Qwen3.5-4B; head L4H6 of pythia-160m; the refusal direction of gemma-2b-it | a gene or variant ([Ensembl](https://www.ensembl.org/), [dbSNP](https://www.ncbi.nlm.nih.gov/snp/)) |
| annotation | a claim that one object does something | "L4H6 is an induction head"; "these two neurons push Qwen toward evil" | a GO or [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) annotation: function, with an evidence code |
| mechanism | a claim about how a computation works, across objects | "refusal is mediated by a single direction"; "induction heads form in a phase change" | a pathway ([Reactome](https://reactome.org/)) |

## Install and run

Python 3.11+.

```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install -e '.[report,test,ingest]'
claimci serve                                # http://127.0.0.1:8765
```

The results below are committed, so the demo needs no keys and no GPU. Adding a source needs a
Claude API key; rerunning claims needs a GPU ([GUIDE.md](GUIDE.md#setup)).

## What it does

1. **Paper to tests on several models.** Extracts a paper's claims and its own tests, checks every
   quote and number against the text, turns them into experiments when the method matches a
   template (steering directions today), and runs them on models the paper did not test.
2. **Everything in local YAML.** Claims, code versions, data provenance and our own choices, plus
   an append-only ledger of results. The web app reads it live.
3. **Re-verification from the metadata alone.** No re-parsing; anything stale reruns, verified
   claims first, cheapest first. A refactor that computes the same thing invalidates nothing.
4. **Results agents can read.** `claimci search / get / card --json`: verdicts with 95% intervals,
   margins, controls, caveats, and whether a threshold was set after seeing results.

Details in [GUIDE.md](GUIDE.md).

## Results

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

\* the steered models lose coherence (5% and 3% of answers coherent); the sources claim the trait,
not coherence, so it is a caveat, not part of the verdict. Every cell opens in the web app to its
evidence: quotes, the authors' tests, our thresholds and who set them, controls, judged samples.

MIT license. [GUIDE.md](GUIDE.md) is the user guide, [SPEC.md](SPEC.md) the formats,
[DESIGN.md](DESIGN.md) where this is going.
