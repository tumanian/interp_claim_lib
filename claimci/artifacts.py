"""Content-addressed tensors that claims pass to each other (e.g. a direction vector).

A binding like {"direction": "artifact:3f2a..."} names a file under ledger/artifacts/.
The id is the sha256 of the saved tensor, so a binding is checkable on any machine.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(self, t: torch.Tensor) -> str:
        buf = io.BytesIO()
        torch.save(t.detach().cpu().contiguous(), buf)
        data = buf.getvalue()
        key = hashlib.sha256(data).hexdigest()[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.pt").write_bytes(data)
        return f"artifact:{key}"

    def save_json(self, obj) -> str:
        """Content-addressed JSON (e.g. judged answers), same reference scheme."""
        import json
        data = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
        key = hashlib.sha256(data).hexdigest()[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_bytes(data)
        return f"artifact:{key}"

    def load(self, ref: str) -> torch.Tensor:
        if not ref.startswith("artifact:"):
            raise ValueError(f"not an artifact reference: {ref!r}")
        return torch.load(self.root / f"{ref.split(':', 1)[1]}.pt", weights_only=True)
