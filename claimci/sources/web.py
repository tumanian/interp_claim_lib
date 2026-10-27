"""Web pages: blogs, LessWrong / Alignment Forum posts, transformer-circuits.pub, personal sites.

Findings come from the page's own summary when it has one (a TL;DR, Summary, Abstract or Key
findings section, in that order), else from the opening of the text; evidence from the whole
page. Title, authors and date come from the page's meta tags (OpenGraph, citation_*, article:*).
"""
from __future__ import annotations

import urllib.request

from .document import Document, html_to_text, section

SUMMARY_HEADINGS = ("TL;DR", "TLDR", "Summary", "Abstract", "Key findings", "Key results", "Overview", "Introduction")
OPENING_CHARS = 3000


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "claimci (+https://github.com/tumanian/interp_claim_lib)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def from_html(url: str, page: str) -> Document:
    text, meta, links, title = html_to_text(page)
    head, summary = section(text, SUMMARY_HEADINGS)
    authors = [a for a in (meta.get("citation_author"), meta.get("author"), meta.get("article:author")) if a]
    date = (meta.get("article:published_time") or meta.get("citation_publication_date") or meta.get("date") or "")[:10]
    return Document(ref=url, kind="blog", title=meta.get("og:title") or meta.get("citation_title") or title,
                    authors=list(dict.fromkeys(authors)), date=date.replace("/", "-"),
                    claims_text=summary or text[:OPENING_CHARS], claims_locator=head or "", body=text, links=links)


def fetch(url: str) -> Document:
    return from_html(url, _get(url))
