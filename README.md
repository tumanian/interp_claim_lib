# claimci \[ [ read only web demo here](https://tumanian.dev/interp_lib_demo/)\]

As mechinterp grows as a field the speed of knowledge discovery needs to match the speed of the model development, so that every time a new model, code or dataset comes up we can verify whether the previous findings hold. To that end we need a database of findings and machinery to replicate findings at scale.  Inspired by the [Goodfire Proposal](https://www.goodfire.com/blog/accelerating-interpretability) and [Gene Ontology](https://geneontology.org/) project, I built a minimal exchange format and finding replication pipeline. 

Where the Gene Ontology annotates a gene product (a protein, an RNA) with a term (a function, process or location) and an evidence code, claimci annotates a model component (a head, a layer, a direction, a neuron) with a statement (what it does, with a metric and threshold) and an evidence code; a claim file is one such annotation.  What's great about the interp vs biology is that we can directly replicate and verify any claim in a paper (or  a tweet(!)) and check the transferability of the claims across models, data sources or even code versions. 

## The demo implements the following: 

1. Given a paper, extracts the claims, formulates experiments to run, runs on different models (expands on what was in the paper if possible)
2. Can rerun/re-verify the claims for new models, code or datasets - this is where most of the compute spend would go to keep the knowledge up to date. We can probably prioritize which claims are more important to recheck regularly.  
3. Stores the metadata (code + set up + anything else) in a local yaml (trivially can be transformed into  a live service)
4. Downstream agents can refer to semi-machine readable results, generate new claims, verify them using the same tools and write the results back. 

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
Formats: [SPEC.md](SPEC.md).

Example results on expansion and re-verification 
| paper | paper's model | expanded to | claims on the expanded model |
|---|---|---|---|
| [Chen et al. 2025, Persona Vectors](https://arxiv.org/abs/2507.21509) | Qwen2.5-7B-Instruct | gemma-2-2b-it | 2 of 2: steering transfers (+37 judged evil; random direction +0.1); monitoring holds (AUROC 1.00) |
| | | Qwen3.5-4B | 2 of 2: steering transfers, weaker (+29 vs +64 on the paper's model); monitoring holds (AUROC 1.00) |
| [Arditi et al. 2024, Refusal direction](https://arxiv.org/abs/2406.11717) | gemma-2b-it | gemma-2-2b-it | 3 of 3: ablation transfers (refusal 0.91 → 0.16; random 0.97), addition transfers (0.00 → 0.99), capabilities kept |
| | | Qwen2.5-7B-Instruct | 2 of 3: ablation transfers (refusal → 0.00), addition transfers, weaker (0.46); capabilities not kept (+0.31 nats) |
| | | Qwen3.5-4B | 2 of 3: addition transfers, weaker (0.24), capabilities kept; ablation didn't transfer (refusal stays 0.85) |
| [Tigges et al. 2023, Linear sentiment](https://arxiv.org/abs/2310.15154) | pythia-1.4b | pythia-2.8b | 3 of 5: ablation transfers (100% of above-chance SST accuracy lost), the direction reads SST (0.77); held-out adjectives and commas-only failed |
| | | pythia-410m | 2 of 5: ablation transfers (87% lost); reads SST 0.76 short of 0.7 at its interval's edge; held-out adjectives and commas-only failed |
| | | gpt2 | 0 of 5: zero-shot SST at chance (0.52), so the ablation tests cannot be measured |
| [Panickssery et al. 2023, CAA](https://arxiv.org/abs/2312.06681) | Llama-2-7b-chat (gated; not run) | Qwen3.5-4B | 1 of 4: corrigibility up transfers (+0.12; control +0.05); corrigibility down and sycophancy both ways did not |
| | | gemma-2-2b-it | 1 of 4: corrigibility up transfers (+0.18); the rest did not |
| | | Qwen2.5-7B-Instruct | 0 of 4: no steering effect observed beyond the random control (sycophancy +0.001 / −0.007, corrigibility +0.024 / −0.033) |
| | | gemma-2b-it | 0 of 4: no steering effect observed beyond the random control |
| [Olsson et al. 2022, induction heads](https://arxiv.org/abs/2209.11895) | in-house models (not public) | pythia-160m, 19 checkpoints | 3 of 3: previous-token head, induction head, ablation (+0.92 nats); forms between step512 and step1000 |



How to install and run on local (read-only):  
```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install -e '.[report,test,ingest]'
claimci serve                                   # http://127.0.0.1:8765
```

To support papers ingestion (needs Claude API keys)
```
cp .env.example .env                            # then set ANTHROPIC_API_KEY=... in .env
claimci ingest arXiv:2310.15154                 # findings from the abstract, the paper's own tests from the body
claimci serve                                   # or use "Add a study" in the web app
```

To support verification experiments (needs GPU box to ssh into)
```
# a GPU box you can SSH into (we used an AWS g6.xlarge); once, on the box: clone, install, add .env
echo '{"host": "ubuntu@<ip>", "key": "~/.ssh/<key>.pem", "dir": "~/interp_claim_lib"}' > .claimci-remote.json
claimci verify --claims sent.pythia14b.ablate   # runs on the box, results merge into ledger/
claimci reverify --dry-run                      # what is stale, in the order it would rerun
claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct --verify
```

To run agents from Claude reading the results (runs from claude)
```
claude                                          # Claude Code in the repo: CLAUDE.md tells it the vocabulary, commands and rules
# then ask, e.g.: "Which steering claims hold on Qwen, and with what margin?"
#                 "Add a verification of src.chen2025.deployment_monitoring on Qwen3.5-4B and run it."
claimci search "steering" --json                # what Claude runs underneath: find claims
claimci get dir.qwen257binstruct.evil.chen2025.steer_up   # verdict, value, 95% interval, margin, controls, caveats
```
Next steps: 

* Batch and scale the verification, to speed up parallel runs of the verification.
* stand up working web UI with nightly verification checks running on the backend
* seed more papers and findings

More: [GUIDE.md](GUIDE.md) (setup, commands, results in detail), [SPEC.md](SPEC.md) (formats),
[DESIGN.md](DESIGN.md) (where this is going). MIT license.
