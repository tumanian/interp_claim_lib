"""Reports from the ledger: validity matrix, throughput, status. No model loads."""
from __future__ import annotations

import re
from collections import Counter

from .ledger import Ledger
from .spec import Repo, Result, evaluate


def revision_order(rev: str) -> tuple:
    m = re.fullmatch(r"step(\d+)", rev)
    return (0, int(m.group(1)), rev) if m else (1, 0, rev)


def current_row(repo: Repo, ledger: Ledger, cid: str, rev: str) -> Result | None:
    """Latest row (any status) for the claim's current key at this revision."""
    rows = ledger.index.get((cid, repo.claim_key(cid, rev)), [])
    return rows[-1] if rows else None


def gated_by(repo: Repo, ledger: Ledger, cid: str, rev: str) -> list[str]:
    """Ancestors that do not hold at this revision under their current expect.

    Values are always recorded; whether a dependent counts is decided here, on read,
    so editing a parent's threshold re-gates without rerunning."""
    out = []
    for a in sorted(repo.ancestors(cid)):
        r = current_row(repo, ledger, a, rev)
        if r is None or r.status != "ok" or evaluate(repo.claims[a].expect, r.value) is False:
            out.append(a)
    return out


def cell(repo: Repo, r: Result | None, gated: list[str] | None = None) -> tuple[str, str]:
    """(text, kind) where kind is pass | fail | gated | other | missing. holds uses the current expect."""
    if r is None:
        return "—", "missing"
    if r.status != "ok":
        return r.status, "other"
    if r.value is not None and repo.claims[r.claim].expect is None:
        return f"{r.value:.3f}", "other"  # a measurement: a value, no verdict
    if gated:
        return f"{r.value:.3f} gated", "gated"
    holds = evaluate(repo.claims[r.claim].expect, r.value)
    return f"{r.value:.3f} {'✓' if holds else '✗'}", "pass" if holds else "fail"


def matrix(repo: Repo, ledger: Ledger, question: str):
    """(claim ids, revisions, cells[claim][revision]) for one question."""
    claims = [c for c in repo.questions[question].claims if c in repo.claims]
    revs = {r.revision for r in ledger.rows
            if r.claim in claims and r.claim_key == repo.claim_key(r.claim, r.revision)}
    revs = sorted(revs, key=revision_order)
    cells = {c: {rev: cell(repo, current_row(repo, ledger, c, rev), gated_by(repo, ledger, c, rev))
                 for rev in revs} for c in claims}
    return claims, revs, cells


def matrix_markdown(repo: Repo, ledger: Ledger, question: str) -> str:
    claims, revs, cells = matrix(repo, ledger, question)
    q = repo.questions[question]
    if not revs:
        return f"**{q.id}**: {q.text}\n\n(no results in the ledger for the current claim keys)"
    lines = [f"**{q.id}**: {q.text}", "", "| claim | evidence | " + " | ".join(revs) + " |",
             "|---|---|" + "---|" * len(revs)]
    for c in claims:
        lines.append(f"| {c} | {repo.claims[c].evidence} | "
                     + " | ".join(cells[c][r][0] for r in revs) + " |")
    return "\n".join(lines)


def matrix_png(repo: Repo, ledger: Ledger, question: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    claims, revs, cells = matrix(repo, ledger, question)
    colors = {"pass": "#cfe8cf", "fail": "#f3c9c4", "gated": "#e6e6e6", "other": "#e6e6e6", "missing": "#ffffff"}
    text = [[cells[c][r][0].replace("✓", "pass").replace("✗", "fail") for r in revs] for c in claims]
    fill = [[colors[cells[c][r][1]] for r in revs] for c in claims]
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * max(len(revs), 1) + 2.6, 0.5 + 0.45 * (len(claims) + 1)))
    ax.axis("off")
    if revs:
        t = ax.table(cellText=text, cellColours=fill, rowLabels=claims, colLabels=revs, loc="center",
                     cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.4)
    ax.set_title(f"{question}: {repo.questions[question].text}", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def throughput(ledger: Ledger) -> list[dict]:
    """One row per mode from that mode's latest run: claims, passes, seconds, claims/minute."""
    latest: dict[str, str] = {}
    for r in ledger.rows:
        if r.cost.get("mode"):
            latest[r.cost["mode"]] = r.run.get("run_id")
    out = []
    for mode in ("shared", "naive"):
        if mode not in latest:
            continue
        rows = [r for r in ledger.rows if r.run.get("run_id") == latest[mode]]
        secs = sum(r.cost.get("seconds", 0.0) for r in rows)
        n = sum(r.status == "ok" for r in rows)
        out.append({"mode": mode, "run": latest[mode], "claims": n,
                    "passes": round(sum(r.cost.get("forward_passes", 0.0) for r in rows), 6),
                    "seconds": round(secs, 1), "per_min": round(60 * n / secs, 1) if secs else None})
    return out


def throughput_markdown(ledger: Ledger) -> str:
    rows = throughput(ledger)
    lines = ["| mode | run | claims verified | forward passes | seconds | claims/minute |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['mode']} | {r['run']} | {r['claims']} | {r['passes']:g} | {r['seconds']} | {r['per_min']} |"
              for r in rows]
    by = {r["mode"]: r for r in rows}
    if "shared" in by and "naive" in by and by["shared"]["passes"]:
        lines.append(f"\nforward passes naive/shared: {by['naive']['passes'] / by['shared']['passes']:.2f}x")
    return "\n".join(lines)


def computed_status(repo: Repo, r: Result | None, ledger: Ledger | None = None) -> str:
    if r is None:
        return "stale"
    if r.status != "ok":
        return r.status
    if repo.claims[r.claim].expect is None:
        return "measured"
    if ledger is not None and gated_by(repo, ledger, r.claim, r.revision):
        return "gated"
    return "verified" if evaluate(repo.claims[r.claim].expect, r.value) else "not holding"


def status_report(repo: Repo, ledger: Ledger, revisions: list[str] | None = None) -> str:
    pairs = [(c, rev) for c in sorted(k for k in repo.claims if not repo.claims[k].is_source)
             for rev in (revisions or [repo.claims[c].revision])]
    rows = {p: current_row(repo, ledger, *p) for p in pairs}
    by_status = Counter(computed_status(repo, rows[p], ledger) for p in pairs)
    by_code = Counter(repo.claims[c].evidence for c in repo.claims)
    lines = ["status:   " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
             "evidence: " + "  ".join(f"{k}={v}" for k, v in sorted(by_code.items()))]
    stale = [(c, rev) for c, rev in pairs if ledger.get(c, repo.claim_key(c, rev)) is None]
    lines.append(f"stale: {len(stale)}")
    for c, rev in stale:
        r = rows[(c, rev)]
        why = f"last run {r.status}: {r.reason}" if r else ledger.stale_reason(repo, c, rev)
        lines.append(f"  {rev:<10} {c:<32} {why}")
    return "\n".join(lines)
