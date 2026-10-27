# Verification on real weights

The cloud build environment could not reach huggingface.co; mechanics were first
tested there on a random-init GPT-NeoX with pythia-70m's architecture
(`tests/models.py`). Everything below was then run on an 8-core Apple Silicon
Mac (CPU, Python 3.11, torch 2.14.1, transformers 5.21.0) with HF access.

Warm the cache once (tests never download):

    python -c "from transformers import AutoModelForCausalLM as M; [M.from_pretrained('EleutherAI/pythia-70m', revision=r) for r in ('main', 'step0')]"

The committed claim files name `pythia-160m` (step 8). Tests copy them into a temp
dir pinned to `pythia-70m` (`tests/models.py:copy_claims`).

## Step 3: backend
- [x] bitwise-identical repeat runs, hook cleanup on exception, hook-leak test (synthetic and real pythia-70m)
- [x] peak RSS < 2 GB through `CpuBackend.load`: 889 MB on real pythia-70m (macOS);
      1199-1450 MB on saved synthetic weights in the cloud env.

## Step 4: experiments (pythia-70m)
- [x] main: prev_token 0.9101 (L2H1), induction 0.6987 (L3H6), ablation loss delta 0.8478 nats
- [x] step0: prev_token 0.0881, induction 0.0360 (both < 0.15)
- [x] attention rows sum to 1, no attention to future positions

## Step 5: ledger
- [x] all checks pass (no model needed)

## Step 6: planner and runner (pythia-70m@main)
- [x] `claimci verify --revisions main`: 3 rows, all hold, 2 forward passes
- [x] rerun: `executed: 0   ledger hits: 3`
- [x] `--mode naive`: values equal shared to 6 decimals; 4 forward passes
      (3 clean, one per claim, plus the ablation's hooked pass)

## Step 7: reports
- [x] matrix (one column `main`, three rows), PNG, throughput (shared 2 < naive 4 passes),
      status `stale: 0`

## Step 8: pythia-160m across eight checkpoints
- [x] `scripts/step8.sh`, checkpoints pre-downloaded (download alone took ~3 min):
      shared 27 s, naive 37 s wall clock; all `check_step8.py` criteria PASS.
      Matrix and throughput in README.md.
- [x] early checkpoints step1..step512 and `main`: the induction head forms between
      step512 and step1000 (ungated values in README.md).
- [x] promotion: all three claims `verified` at `main` with `basis: supplemented_expect`.
      Thresholds are ours (PROMPT.md), recorded under `provenance.supplemented.expect`.
      The ablation claim's `stated_value: 0.4` / `locator` were placeholders copied
      from the SPEC example and have been removed.
- Open: blocked claims record no value, so the matrix shows `blocked` where the
  ungated induction score is ~0.04. Exposing values for blocked claims would show the
  onset in the matrix itself.

## Step 9: README
- [x] fresh clone + venv + `pip install -e '.[report,test]'` + `claimci validate && claimci plan --revisions main`
- [x] step 8 matrix and throughput table in README.md
- [ ] the optional CPU-wheel line (`download.pytorch.org`) is untested (the macOS default wheel is CPU already)
