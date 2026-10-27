"""What keeps one claim from spoiling another: a per-claim time limit, and the contamination canary
(a hook that outlived its intervention would silently change every later claim on the same model)."""
from __future__ import annotations

import signal
from contextlib import contextmanager

CANARY_N = 4
CANARY_ATOL = 1e-4


class ClaimTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float | None):
    """Per-claim wall clock, via SIGALRM where available (main thread on Unix)."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def fire(signum, frame):
        raise ClaimTimeout(f"claim exceeded {seconds}s")

    try:
        old = signal.signal(signal.SIGALRM, fire)
    except ValueError:  # not the main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def contamination(backend, handle, ds, acts) -> str | None:
    """Why the model no longer computes what the shared pass did, or None: after an intervention,
    a cheap probe on the first inputs must match the shared pass."""
    import torch
    now, before = backend.probe(handle, ds), backend.probe_of(acts)
    # bf16 on a GPU is not bitwise stable across batch shapes; a leaked hook moves far more than this
    atol = CANARY_ATOL if getattr(handle, "opts", {}).get("dtype", "float32") == "float32" else 5e-2
    if torch.allclose(now, before, atol=atol):
        return None
    return f"canary probe on the first {CANARY_N} inputs differs from the shared pass by {(now - before).abs().max().item():.6f}"
