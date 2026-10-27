"""Deterministic dataset generators and content hashing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schema.claim import DataVersion


def repeated_random_tokens(n: int, half: int, seed: int, vocab_lo: int, vocab_hi: int) -> list[list[int]]:
    """n sequences of length 2*half; tokens uniform in [vocab_lo, vocab_hi); second half repeats the first."""
    rng = np.random.Generator(np.random.PCG64(seed))
    first = rng.integers(vocab_lo, vocab_hi, size=(n, half), dtype=np.int64)
    return np.concatenate([first, first], axis=1).tolist()


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def vendored(files: dict) -> dict:
    """Named files under data/: JSON is loaded; anything else is referenced by path and sha256."""
    out = {}
    for name, rel in sorted(files.items()):
        p = DATA_DIR / rel
        if p.suffix == ".json":
            out[name] = json.loads(p.read_text())
        else:
            out[name] = {"file": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


GENERATORS = {"repeated_random_tokens": repeated_random_tokens, "vendored": vendored}


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
    tokens: list[list[int]] | dict  # token sequences, or named records for prompt datasets

    @property
    def is_prompts(self) -> bool:
        return isinstance(self.tokens, dict)

    def head(self, k: int) -> "Dataset":
        return Dataset(self.version, self.tokens[:k])

    @property
    def meta(self) -> dict:
        v = self.version
        return {"gen": v.gen, "params": dict(v.params), "hash": v.hash, "n": len(self.tokens)}


def load_dataset(dv: DataVersion) -> Dataset:
    return Dataset(dv, materialize(dv))
