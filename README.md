# claimci

If interp is to grow as a field the speed of knowledge discovery needs to match the speed of the model development, so that every time a new model, code or dataset comes up we can verify whether the previous findings hold. To that end we need a database of findings and machinery to replicate findings at scale.  Inspired by the [Goodfire Proposal](<GF post link>) and [Gene Onthology](https://geneontology.org/) project, I built a minimal exchange format and finding replication pipeline. The mapping from Gene Ontology to interp can be: 

| Gene Ontology | claimci |
|---|---|
| Gene product (a protein, an RNA) | Model component (a head, a layer, a direction, a neuron) |
| GO term (a function, process or location) | Statement (what the component does, with a metric and threshold) |
| Annotation: gene product × term × evidence | Claim file: subject × statement × evidence |

Whats great about the interp field  vs biology is that we can directly replicate and verify any claim in a paper (or  a tweet :D ) and check the transferability of the claims across models, data sources or even code versions. 

The demo below shows the following: 

1. Given a paper, extracts the claims, formulates experiments to run, runs on different models (expands on what was in the paper if possible)
2. Can rerun/re-verfy the claims for new models, code or datasets - this is where most of the compute spend would go to keep the knowledge up to date. We can probably pririize which claims are more important to recheck regularly.  
3. Stores the metadata (code + set up + anything else) in a local yaml (trivially can be transformed into  a live service)
4. Downstream agents can refer to semi-machine readable results, generate new claims, verify them using the same tools and write the results back. 


The read-only demo for the verifications and expansions (where we try the method on a new model) that I ran [here](https://tumanian.dev/interp_lib_demo/)

To run the  paper ingestion and verification pipeline you will need 

* Claude API keys for ingestion, claim extraction
* AWS keys for running the verification experiments 


Next steps: 

* Batch and scale the verification, to speed up parallel runs of the verifcation. 



Ingestion and verification tested on so far: 
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al. 2024): 4 of 5 findings verified on gemma-2b-it
- [Persona Vectors](https://arxiv.org/abs/2507.21509) (Chen et al. 2025): 2 of 5 verified on Qwen2.5-7B-Instruct; expanded to gemma-2b-it (could not run)
- [Linear Representations of Sentiment in Large Language Models](https://arxiv.org/abs/2310.15154) (Tigges et al. 2023): 2 of 4 verified on pythia-1.4b, 2 not reproduced
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Panickssery et al. 2023): expanded to Qwen2.5-7B-Instruct and Qwen3.5-4B, where it does not transfer; not run on Llama-2 (gated)
- [@_lyraaaa_ on X](https://x.com/_lyraaaa_/status/2113833330429083662): two neurons make Qwen3.5-4B evil, verified (it loses coherence)
- Induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)): form between step512 and step1000 of pythia-160m, verified

How to install and run on local (readonly ):  
```
git clone https://github.com/tumanian/interp_claim_lib && cd interp_claim_lib
python -m venv .venv && . .venv/bin/activate
pip install -e '.[report,test,ingest]'
claimci serve                                   # http://127.0.0.1:8765
```

To support papers ingestion
```
cp .env.example .env                            # then set ANTHROPIC_API_KEY=... in .env
claimci ingest arXiv:2310.15154                 # findings from the abstract, the paper's own tests from the body
claimci serve                                   # or use "Add a source" in the web app
```

To support verification experiments
```
# a GPU box you can SSH into (we used an AWS g6.xlarge); once, on the box: clone, install, add .env
echo '{"host": "ubuntu@<ip>", "key": "~/.ssh/<key>.pem", "dir": "~/interp_claim_lib"}' > .claimci-remote.json
claimci verify --claims sent.pythia14b.ablate   # runs on the box, results merge into ledger/
claimci reverify --dry-run                      # what is stale, in the order it would rerun
claimci expand arXiv:2312.06681 --models Qwen/Qwen2.5-7B-Instruct --verify
```

To run agents from Claue reading the results 
```
claude                                          # Claude Code in the repo: CLAUDE.md tells it the vocabulary, commands and rules
# then ask, e.g.: "Which steering claims hold on Qwen, and with what margin?"
#                 "Add a verification of src.chen2025.deployment_monitoring on Qwen3.5-4B and run it."
claimci search "steering" --json                # what Claude runs underneath: find claims
claimci get dir.qwen257binstruct.evil.chen2025.steer_up   # verdict, value, 95% interval, margin, controls, caveats
```

More: [GUIDE.md](GUIDE.md) (setup, commands, results in detail), [SPEC.md](SPEC.md) (formats),
[DESIGN.md](DESIGN.md) (where this is going). MIT license.
