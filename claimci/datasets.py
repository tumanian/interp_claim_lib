"""Deterministic dataset generators and content hashing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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


@dataclass
class Dataset:
    version: DataVersion
    tokens: list[list[int]]

    def head(self, k: int) -> "Dataset":
        return Dataset(self.version, self.tokens[:k])

    @property
    def meta(self) -> dict:
        v = self.version
        return {"gen": v.gen, "params": dict(v.params), "hash": v.hash, "n": len(self.tokens)}


def load_dataset(dv: DataVersion) -> Dataset:
    return Dataset(dv, materialize(dv))
