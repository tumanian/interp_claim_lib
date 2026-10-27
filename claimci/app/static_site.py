"""The web app as a static site: every page it reads, pre-rendered to JSON next to index.html.

For a read-only demo on any static host (Vercel, GitHub Pages): no server, so nothing can call the
Claude API or a GPU box. Adding sources and running verifications show as unavailable.

    claimci export-site --out site                         # then: npx vercel deploy site
    claimci export-site --out ../blog/static/demo --base /demo/   # served under a path of a site
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..read import cards
from ..schema.loader import load_repo
from ..store.ledger import Ledger
from .web import PAGE, overview, samples

READ_ONLY = {"ingest": "Read-only demo: adding a source runs locally with a Claude API key (see GUIDE.md).",
             "verify": "Read-only demo: verifications run locally on a GPU (see GUIDE.md)."}


def key(request: str) -> str:
    """The file a request is saved as; index.html computes the same name in static mode."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", request) + ".json"


def export(root: Path, out: Path, base: str | None = None, verified_only: bool = False) -> int:
    """Write the site to `out`. `base` is the URL path it will be served under (e.g. /demo/), so
    its relative requests resolve with or without a trailing slash. `verified_only` leaves out
    sources none of whose findings is verified yet."""
    root, out = Path(root), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "api").mkdir(parents=True)
    repo = load_repo(root)
    ledger = Ledger.for_repo(repo)
    pages = {"overview": {**overview(root), "why_not": READ_ONLY}}
    if verified_only:
        pages["overview"]["sources"] = [s for s in pages["overview"]["sources"]
                                        if any(c["support"] == "supported" for c in s["claims"])]
    for s in pages["overview"]["sources"]:
        pages[f"report?ref={s['ref']}"] = cards.report(repo, ledger, s["ref"])
    for art in sorted((root / "ledger" / "artifacts").glob("*.json")):
        ref = f"artifact:{art.stem}"
        try:
            pages[f"samples?ref={ref}"] = samples(root, ref)
        except Exception:   # not a list of judged answers
            continue
    for request, body in pages.items():
        (out / "api" / key(request)).write_text(json.dumps(body, ensure_ascii=False, default=str))
    head = "<head>\n" + (f'<base href="{base}">\n' if base else "") + "<script>window.CLAIMCI_STATIC = true;</script>"
    page = PAGE.read_text().replace("<head>", head, 1)
    (out / "index.html").write_text(page)
    return len(pages)
