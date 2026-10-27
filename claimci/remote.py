"""Run `verify` on a remote GPU box over SSH and merge its ledger back.

Configured by .claimci-remote.json at the repo root (not committed):
    {"host": "ubuntu@1.2.3.4", "key": "~/.ssh/key.pem", "dir": "~/interp_claim_lib"}
The remote needs the repo's Python environment (.venv) set up once. The ledger is
append-only JSONL, so merging is a union of rows; artifacts are content-addressed files.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PUSH = ["claimci/", "claims/", "questions/", "data/", "pyproject.toml"]


def config(root: Path) -> dict | None:
    p = Path(root) / ".claimci-remote.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ssh(cfg: dict) -> list[str]:
    return ["ssh", "-i", os.path.expanduser(cfg["key"]), "-o", "BatchMode=yes", cfg["host"]]


def _rsync(cfg: dict) -> list[str]:
    return ["rsync", "-az", "--exclude", "__pycache__", "-e", f"ssh -i {os.path.expanduser(cfg['key'])} -o BatchMode=yes"]


def merge_ledger(local: Path, remote_copy: Path) -> int:
    """Append rows present remotely but not locally; returns how many were added."""
    have = set(local.read_text().splitlines()) if local.exists() else set()
    new = [l for l in remote_copy.read_text().splitlines() if l.strip() and l not in have]
    if new:
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("a") as f:
            f.write("\n".join(new) + "\n")
    return len(new)


def verify(root: Path, claims: list[str], log=print, force: bool = False) -> int:
    root = Path(root)
    cfg = config(root)
    if cfg is None:
        raise RuntimeError("no .claimci-remote.json; see claimci/remote.py")
    d = cfg["dir"]
    for item in PUSH:  # a trailing slash syncs a directory's contents (Path() would strip it)
        src = str(root / item) + ("/" if item.endswith("/") else "")
        subprocess.run(_rsync(cfg) + [src, f"{cfg['host']}:{d}/{item}"], check=True)
    cmd = f"cd {d} && . .venv/bin/activate && set -a && {{ [ -f .env ] && . ./.env; true; }} && set +a && PYTHONUNBUFFERED=1 claimci verify --claims {','.join(claims)}" \
          + (" --force" if force else "")
    log(f"remote: {cfg['host']}: claimci verify --claims {','.join(claims)}")
    proc = subprocess.Popen(_ssh(cfg) + [cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line and not any(s in line for s in ("causal_conv1d", "Loading weights", "Warning", "warn(")):
            log(line)
    rc = proc.wait()
    (root / "ledger").mkdir(exist_ok=True)
    tmp = root / "ledger" / ".remote_results.jsonl"
    subprocess.run(_rsync(cfg) + [f"{cfg['host']}:{d}/ledger/results.jsonl", str(tmp)], check=True)
    subprocess.run(_rsync(cfg) + [f"{cfg['host']}:{d}/ledger/artifacts/", str(root / "ledger" / "artifacts") + "/"],
                   check=False)
    added = merge_ledger(root / "ledger" / "results.jsonl", tmp)
    tmp.unlink()
    log(f"merged {added} ledger rows from the remote (exit code {rc})")
    return rc
