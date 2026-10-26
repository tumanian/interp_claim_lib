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

## Step 5: ledger
- [x] all checks pass here (no model needed; "full run" rows are written directly by the test)

## Step 6: planner and runner
Verified here on synthetic weights with thresholds lowered in a temp copy
(`tests/test_runner.py`): plan with an empty HF cache and without importing
torch; 3 rows written; rerun = 0 executed / 3 ledger hits; naive values equal
shared values to 6 decimals; failing parent -> dependent `blocked`, no
intervention pass; raising experiment -> `error`, siblings ok; leaked hook ->
canary -> `hook_leak`; two forced runs give identical values.

- [ ] `claimci verify --revisions main` on real pythia-70m: 3 rows, all three hold
      (`pytest tests/test_runner.py::test_real_model_all_three_hold -s` does this in a temp copy)
- [ ] rerun reports `executed: 0   ledger hits: 3`
- [ ] `claimci verify --revisions main --mode naive`: values equal shared to 6 decimals
- Spec note: naive mode records **4** forward passes, not 3. Naive gives each claim its
  own clean pass (3), and the ablation also needs its hooked pass (1). Shared mode is 2.

## Step 7: reports
- [x] matrix (one column, three rows, values match ledger), PNG renders, throughput
      shared < naive passes, status stale=0 then `code changed` for the edited claim
      and its dependent; editing `expect` re-evaluates holds without rerun
      (`tests/test_report.py`, synthetic ledger)
- [ ] same three commands after the real step 6 runs

## Step 8: the real run (not run; needs HF access)
- [ ] `scripts/step8.sh` switches the claim files to pythia-160m, runs shared and naive
      verification over the eight checkpoints, renders `matrix.png`, prints throughput,
      then `scripts/check_step8.py` prints raw values and PASS/FAIL for each criterion:
      wall clock < 15 min, step0 all fail, induction onset in [step1000, step16000] and
      holding after. Commit the claim-file switch and matrix.png afterwards.
- Dry run on synthetic weights: 8 checkpoints in 16 s excluding downloads; the check
  script runs and correctly fails the onset criteria (random weights).
- Note: when `mech.prev_token_head` does not hold at a checkpoint, the induction claims
  there are `blocked` (per RUNNER.md) and have no raw value. If the onset check fails,
  `claimci verify --claims mech.induction_head` won't help either, since ancestors are
  included; we'd need to discuss whether to expose raw values for blocked claims.
