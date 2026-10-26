"""Deterministic dataset generators and content hashing."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .spec import DataVersion


def repeated_random_tokens(n: int, half: int, seed: int, vocab_lo: int, vocab_hi: int) -> list[list[int]]:
    """n sequences of length 2*half; tokens uniform in [vocab_lo, vocab_hi); second half repeats the first."""
    rng = np.random.Generator(np.random.PCG64(seed))
    first = rng.integers(vocab_lo, vocab_hi, size=(n, half), dtype=np.int64)
    return np.concatenate([first, first], axis=1).tolist()


GENERATORS = {"repeated_random_tokens": repeated_random_tokens}


def content_hash(data) -> str:
    """sha256 of the canonical JSON of the materialized data."""
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def materialize(dv: DataVersion) -> list[list[int]]:
    if dv.gen not in GENERATORS:
        raise KeyError(f"unknown dataset generator {dv.gen!r}")
    return GENERATORS[dv.gen](**dv.params)
