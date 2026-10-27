"""One-time migration to stable experiment ids and behaviour hashes (DESIGN.md section 14).

    python scripts/migrate_keys.py OLD_KEYS.json [--ledger ledger/results.jsonl]

1. Rewrites claim files: `entrypoint: <python path>` becomes `experiment: <registered id>`, and the
   now meaningless `version: local` line goes. Text edits, so comments survive.
2. For every claim@revision whose current row (under the old key, from OLD_KEYS.json, computed with
   the old code) exists, appends a copy under the new key with `migrated_from: <old key>` and the new
   code version and lineage. Append-only, nothing recomputed: valid because only decorators and
   import lines changed in the experiments, which the behaviour hash ignores anyway.

OLD_KEYS.json maps "claim_id@revision" to the claim key computed before the code changed.
"""
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimci.schema.codehash import by_entrypoint, clear_cache  # noqa: E402
from claimci.store.ledger import Ledger  # noqa: E402
from claimci.schema.loader import load_repo  # noqa: E402


def rewrite_claim_files() -> int:
    n = 0
    for p in sorted((ROOT / "claims").glob("*.yaml")):
        text = p.read_text()
        m = re.search(r"(?m)^(\s*)entrypoint:\s*(\S+)\s*$", text)
        if not m:
            continue
        reg = by_entrypoint(m.group(2))
        if reg is None:
            raise SystemExit(f"{p.name}: {m.group(2)} is not a registered experiment")
        text = text[:m.start()] + f"{m.group(1)}experiment: {reg.id}" + text[m.end():]
        text = re.sub(r"(?m)^\s*version:\s*local\s*\n", "", text)
        p.write_text(text)
        n += 1
    return n


def migrate_ledger(old_keys: dict, ledger_path: Path) -> tuple[int, int]:
    clear_cache()
    repo = load_repo(ROOT)
    led = Ledger(ledger_path)
    added = skipped = 0
    for k, old in sorted(old_keys.items()):
        cid, rev = k.rsplit("@", 1)
        if cid not in repo.claims:
            continue
        rows = led.index.get((cid, old), [])
        if not rows:
            continue
        new = repo.claim_key(cid, rev)
        if new == old or led.index.get((cid, new)):
            skipped += 1
            continue
        latest = rows[-1]
        led.put(replace(latest, claim_key=new, code_version=repo.claims[cid].code_version,
                        lineage=repo.lineage(cid), migrated_from=old))
        added += 1
    return added, skipped


if __name__ == "__main__":
    old_keys = json.loads(Path(sys.argv[1]).read_text())
    ledger = Path(sys.argv[sys.argv.index("--ledger") + 1]) if "--ledger" in sys.argv else ROOT / "ledger" / "results.jsonl"
    print(f"rewrote {rewrite_claim_files()} claim files")
    added, skipped = migrate_ledger(old_keys, ledger)
    print(f"appended {added} migrated rows; {skipped} already current")
