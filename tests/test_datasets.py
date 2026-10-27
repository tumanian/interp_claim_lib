import shutil
import subprocess
import sys
from pathlib import Path

from claimci.store.datasets import content_hash, repeated_random_tokens
from claimci.schema.loader import load_repo

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(n=48, half=20, seed=0, vocab_lo=1000, vocab_hi=10000)
HASH_SNIPPET = (
    "from claimci.store.datasets import repeated_random_tokens, content_hash;"
    "print(content_hash(repeated_random_tokens(n=48, half=20, seed=0, vocab_lo=1000, vocab_hi=10000)))"
)


def test_shape_and_repeat():
    data = repeated_random_tokens(**PARAMS)
    assert len(data) == 48
    for seq in data:
        assert len(seq) == 40 and seq[:20] == seq[20:]
        assert all(1000 <= t < 10000 for t in seq)


def test_hash_stable_across_processes():
    hashes = [subprocess.run([sys.executable, "-c", HASH_SNIPPET], capture_output=True, text=True,
                             check=True).stdout.strip() for _ in range(2)]
    print(hashes)
    assert hashes[0] == hashes[1] == content_hash(repeated_random_tokens(**PARAMS))


def test_seed_changes_hash():
    assert content_hash(repeated_random_tokens(**PARAMS)) != \
        content_hash(repeated_random_tokens(**{**PARAMS, "seed": 1}))


def test_declared_hash_matches_generated():
    repo = load_repo(ROOT)
    for cid in ("mech.prev_token_head", "mech.induction_head", "causal.induction_head_ablation"):
        dv = repo.claims[cid].dataset
        assert dv.hash == content_hash(repeated_random_tokens(**dv.params))


def test_validate_fails_on_one_char_hash_edit(tmp_path):
    for d in ("claims", "questions"):
        shutil.copytree(ROOT / d, tmp_path / d)
    p = tmp_path / "claims" / "mech.induction_head.yaml"
    text = p.read_text()
    declared = load_repo(tmp_path).claims["mech.induction_head"].dataset.hash
    flipped = declared[:-1] + ("0" if declared[-1] != "0" else "1")
    p.write_text(text.replace(declared, flipped))
    r = subprocess.run([sys.executable, "-m", "claimci.cli", "--root", str(tmp_path), "validate"],
                       capture_output=True, text=True)
    print(r.stderr)
    assert r.returncode != 0
    assert "mech.induction_head" in r.stderr and "data.hash mismatch" in r.stderr
