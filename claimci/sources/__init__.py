"""Sources: turn a paper, blog post or social post into one Document, then into claims.

Adapters know one kind of source each (how to get its text, where its findings and its evidence
are); extraction (claimci/sources/extract.py) is shared by all of them.

    route("arXiv:2406.11717")                     -> arxiv adapter
    route("https://transformer-circuits.pub/...") -> web adapter (blogs, LessWrong, Alignment Forum)
    route("https://x.com/...", text="...")        -> post adapter (pasted text: X and other walled pages)
"""
from __future__ import annotations

import re

from .document import Document

ARXIV = re.compile(r"^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$|^https?://arxiv\.org/(abs|html|pdf)/", re.I)


def route(source: str, text: str | None = None) -> Document:
    source = source.strip()
    if text is not None:
        from .post import from_text
        return from_text(source, text)
    if ARXIV.match(source):
        from .arxiv import fetch
        return fetch(source)
    if source.startswith(("http://", "https://")):
        from .web import fetch
        return fetch(source)
    raise ValueError(f"not an arXiv id or URL: {source!r} (for a post that cannot be fetched, give its text)")
