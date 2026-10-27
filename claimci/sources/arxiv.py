"""arXiv: metadata and abstract from the API, the body from the HTML rendering."""
from __future__ import annotations

import html
import re
import urllib.request

from .document import Document, github_repos, html_to_text


def arxiv_id(source: str) -> str:
    return re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|html|pdf)/)", "", source.strip(), flags=re.I).strip("/")


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def metadata(aid: str) -> dict:
    """title, abstract, first-version date and authors from the arXiv API, not from a model."""
    entry = _get(f"https://export.arxiv.org/api/query?id_list={aid}").split("<entry>", 1)[-1]
    one = lambda tag: re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if (m := re.search(fr"<{tag}>(.*?)</{tag}>", entry, re.S)) else ""
    date = re.search(r"<published>(\d{4}-\d{2}-\d{2})", entry)
    return {"title": one("title"), "abstract": one("summary"), "date": date.group(1) if date else "",
            "authors": [html.unescape(a.strip()) for a in re.findall(r"<author>\s*<name>(.*?)</name>", entry, re.S)]}


def fetch(source: str) -> Document:
    aid = arxiv_id(source)
    meta = metadata(aid)
    body, _, links, _ = html_to_text(_get(f"https://arxiv.org/html/{aid}"))
    return Document(ref=f"arXiv:{re.sub(r'v[0-9]+$', '', aid)}", kind="paper", title=meta["title"],
                    authors=meta["authors"], date=meta["date"], claims_text=meta["abstract"],
                    claims_locator="Abstract", body=body, links=links, own_code=[])
    # own_code is left to extraction: a paper names many repositories, and only the text says
    # which is the authors' release (github_repos(links) would include everything cited)


__all__ = ["fetch", "metadata", "arxiv_id", "github_repos"]
