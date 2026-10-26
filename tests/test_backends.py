import subprocess
import sys
from pathlib import Path

import pytest
import torch

from claimci.adapters import HFAdapter
from claimci.backends import CpuBackend, registered
from claimci.datasets import load_dataset
from claimci.loader import load_repo
from tests.models import any_handle, smallest_cached, synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    backend = CpuBackend()
    handle, kind = any_handle()
    print(f"\nmodel weights: {kind} ({handle.name}@{handle.revision})")
    ds = load_dataset(load_repo(ROOT).claims["mech.induction_head"].dataset)
    return backend, handle, ds


def zero_input(module, args):
    return (torch.zeros_like(args[0]),) + args[1:]


def test_acts_shapes(setup):
    backend, handle, ds = setup
    a = backend.run(handle, ds)
    L, H, D = handle.n_layers, handle.n_heads, handle.model.config.hidden_size
    assert a.hidden.shape == (L + 1, 48, 40, D)
    assert a.attn.shape == (L, 48, H, 40, 40)
    assert a.loss.shape == (48, 39)
    assert a.lengths.tolist() == [40] * 48
    assert a.meta["hash"] == ds.version.hash


def test_loss_matches_hf_lm_loss(setup):
    backend, handle, ds = setup
    ids = torch.tensor(ds.tokens)
    with torch.no_grad():
        ref = handle.model(input_ids=ids, labels=ids).loss
    got = backend.run(handle, ds).loss.mean()
    print(f"mean loss: claimci {got.item():.6f}  hf {ref.item():.6f}")
    assert torch.allclose(got, ref, atol=1e-5)


def test_runs_are_bitwise_identical(setup):
    backend, handle, ds = setup
    a, b = backend.run(handle, ds), backend.run(handle, ds)
    assert a.equal(b)
    for k in ("hidden", "attn", "loss", "lengths"):
        assert torch.equal(getattr(a, k), getattr(b, k))


def test_raising_hook_leaves_no_handles(setup):
    backend, handle, ds = setup
    module = HFAdapter(handle).attn_out_proj(1)

    def boom(module, args):
        raise RuntimeError("hook failed mid-forward")

    with pytest.raises(RuntimeError, match="mid-forward"):
        backend.run(handle, ds, hooks=[(module, boom)])
    assert len(module._forward_pre_hooks) == 0
    with pytest.raises(ValueError):
        with registered([(module, zero_input)]):
            assert len(module._forward_pre_hooks) == 1
            raise ValueError
    assert len(module._forward_pre_hooks) == 0


def test_hook_changes_loss_and_does_not_leak(setup):
    backend, handle, ds = setup
    module = HFAdapter(handle).attn_out_proj(2)
    clean = backend.run(handle, ds)
    hooked = backend.run(handle, ds, hooks=[(module, zero_input)])
    after = backend.run(handle, ds)
    delta = (hooked.loss - clean.loss).abs().max().item()
    print(f"max |loss change| from zeroing layer-2 attention output: {delta:.6f}")
    assert delta > 1e-4
    assert torch.allclose(after.loss, clean.loss, atol=1e-6)


RSS_SCRIPT = """
import resource, sys
sys.path.insert(0, {root!r})
from claimci.backends import CpuBackend
from claimci.datasets import load_dataset
from claimci.loader import load_repo
from tests.models import MODEL, smallest_cached
rev = smallest_cached()
b = CpuBackend(offline=True)
h = b.load(MODEL, rev) if rev else b.load({saved!r}, "main")
ds = load_dataset(load_repo({root!r}).claims["mech.induction_head"].dataset)
b.run(h, ds)
try:  # VmHWM is per address space; ru_maxrss can carry the forking parent's peak across exec
    kb = next(int(ln.split()[1]) for ln in open("/proc/self/status") if ln.startswith("VmHWM"))
except OSError:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("real" if rev else "synthetic", kb)
"""


def test_peak_rss_under_2gb(tmp_path):
    """Measured in a fresh process through CpuBackend.load (from_pretrained), the path
    real checkpoints take. Without a cached checkpoint, synthetic weights are saved to
    disk first: building a random-init model in-process inflates peak RSS by ~450 MB."""
    if smallest_cached() is None:
        synthetic().model.save_pretrained(tmp_path / "synthetic70m")
    script = RSS_SCRIPT.format(root=str(ROOT), saved=str(tmp_path / "synthetic70m"))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         check=True).stdout.split()
    kind, kb = out[-2], int(out[-1])
    print(f"peak RSS ({kind} weights): {kb / 1024:.0f} MB")
    assert kb < 2 * 1024 * 1024
