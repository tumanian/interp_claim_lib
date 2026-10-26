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
