"""Check step 8 criteria from the ledger and print raw values. Never edits thresholds."""
import sys
from pathlib import Path

from claimci.store.ledger import Ledger
from claimci.schema.loader import load_repo
from claimci.read.report import computed_status, current_row

revs = sys.argv[1].split(",")
wall = int(sys.argv[2])
repo = load_repo(Path(__file__).resolve().parents[1])
ledger = Ledger.for_repo(repo)
ids = repo.questions["q.induction_formation"].claims
status = {c: [computed_status(repo, current_row(repo, ledger, c, r), ledger) for r in revs] for c in ids}
print("\nraw values:")
for c in ids:
    vals = [current_row(repo, ledger, c, r) for r in revs]
    print(f"  {c:<32}", "  ".join(f"{r}={v.value if v and v.value is not None else v.status if v else '-'}"
                                  for r, v in zip(revs, vals)))

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")

check(f"shared wall clock {wall}s < 900s (PROMPT.md step 8; RUNNER.md's target is 600s incl. downloads)", wall < 900)
check("step0: no claim holds (negative control)", all(status[c][0] != "verified" for c in ids))
ind = status["mech.induction_head"]
onset = next((i for i, s in enumerate(ind) if s == "verified"), None)
check(f"induction head onset between step1000 and step16000 (onset: {revs[onset] if onset is not None else 'none'})",
      onset is not None and revs.index("step1000") <= onset <= revs.index("step16000"))
check("induction head holds at every checkpoint after onset",
      onset is not None and all(s == "verified" for s in ind[onset:]))
check("matrix.png exists", Path("matrix.png").exists())
sys.exit(0 if ok else 1)
