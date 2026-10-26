# Pending verification (needs Hugging Face access)

The cloud build environment could not reach huggingface.co, so nothing here
has run on real Pythia weights. Mechanics were tested on a random-init
GPT-NeoX with pythia-70m's architecture, built locally (`tests/models.py`).
Run these on a machine with HF access. Tests that need trained weights skip
with a message until the checkpoint is in the HF cache.

Warm the cache once (tests never download):

    python -c "from transformers import AutoModelForCausalLM as M; [M.from_pretrained('EleutherAI/pythia-70m', revision=r) for r in ('main', 'step0')]"

## Step 3: backend
- [x] bitwise-identical repeat runs, hook cleanup on exception, hook-leak test (synthetic weights)
- [ ] same tests on real pythia-70m: `pytest tests/test_backends.py -s` (uses `main` once cached)
- [ ] peak RSS < 2 GB on real weights. Synthetic measured 1901 MB, of which ~520 MB is
      importing torch+transformers and ~450 MB is building the random-init model. Tight margin.

## Step 4: experiments
- [x] scores and bindings exact on hand-built attention patterns
- [x] attention rows sum to 1 (atol 1e-4), no attention to future positions (synthetic)
- [x] ablation runs one hooked pass and leaves the next clean run unchanged (synthetic)
- [ ] `pytest tests/test_experiments.py -s` with `main` and `step0` cached:
      main: prev_token >= 0.5, induction >= 0.3, ablation delta > 0 (print it; if < 0.3, report, don't tune)
      step0: both scores < 0.15
      For reference, random-init synthetic weights gave prev_token 0.0855, induction 0.0347
      (close to uniform causal attention: ~0.083 and ~0.034). That is not the step0 checkpoint.
