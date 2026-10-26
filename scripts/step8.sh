#!/usr/bin/env bash
# Step 8: pythia-160m across eight training checkpoints. Needs Hugging Face access.
# Switches the three claim files to pythia-160m (commit that change afterwards),
# runs shared then naive verification, renders the matrix, and checks the
# step 8 criteria against the ledger. Thresholds are never edited here.
set -euo pipefail
cd "$(dirname "$0")/.."
REVS=step0,step1000,step2000,step4000,step8000,step16000,step32000,step143000

sed -i.bak 's#EleutherAI/pythia-70m#EleutherAI/pythia-160m#' claims/*.yaml && rm claims/*.yaml.bak
claimci validate

start=$(date +%s)
claimci verify --revisions "$REVS" || shared_rc=$?   # rc 1 = some claim errored; rows are kept
shared_end=$(date +%s)
echo "shared run wall clock (incl. downloads): $((shared_end - start))s"

claimci matrix --question q.induction_formation --png matrix.png
claimci verify --revisions "$REVS" --mode naive || naive_rc=$?
echo "naive run wall clock: $(( $(date +%s) - shared_end ))s"
claimci throughput
echo "verify exit codes: shared ${shared_rc:-0}, naive ${naive_rc:-0} (1 = a claim errored; see ledger)"
python scripts/check_step8.py "$REVS" "$((shared_end - start))"
