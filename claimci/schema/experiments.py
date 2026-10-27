"""The registered experiments as recorded in versions.yaml: read without importing experiment code
(no torch), which is all loading claims needs. Written by claimci/experiments/versions.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

LOCK_PATH = Path(__file__).resolve().parents[1] / "experiments" / "versions.yaml"


@lru_cache(maxsize=None)
def _read(mtime: float) -> dict:
    return yaml.safe_load(LOCK_PATH.read_text()) or {} if LOCK_PATH.exists() else {}


def lock() -> dict:
    """{experiment id: {version, entrypoint, params, tags, fingerprint}}."""
    return _read(LOCK_PATH.stat().st_mtime if LOCK_PATH.exists() else 0.0)


def by_entrypoint(entrypoint: str) -> tuple[str, dict] | None:
    return next(((eid, e) for eid, e in lock().items() if e.get("entrypoint") == entrypoint), None)
