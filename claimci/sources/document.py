"""The one shape every source becomes before extraction, and HTML-to-text for adapters."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Document:
    ref: str                     # stable reference: arXiv:2406.11717, a URL
    kind: str                    # paper | blog | post
    title: str = ""
    authors: list[str] = field(default_factory=list)
    date: str = ""               # YYYY-MM-DD when known
    claims_text: str = ""        # where the source states its findings: abstract, TL;DR, summary, the post
    claims_locator: str = ""     # what that part is called ("Abstract", "TL;DR"); empty when uncitable
    body: str = ""               # where its evidence is: the body, or the whole text
    links: list[str] = field(default_factory=list)
    own_code: list[str] = field(default_factory=list)   # repositories the source says are its own release


class _Text(HTMLParser):
    """HTML to plain text: headings marked with '## ', math replaced by its alt text, links kept."""
    SKIP = {"script", "style", "nav", "header", "footer", "button", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.out, self.skip, self.links, self.meta, self.in_title, self.title = [], 0, [], {}, False, ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and a.get("content"):
            key = a.get("property") or a.get("name")
            if key:
                self.meta.setdefault(key.lower(), a["content"])
        if tag == "a" and a.get("href", "").startswith("http"):
            self.links.append(a["href"])
        if tag == "title":
            self.in_title = True
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5"):
            self.out.append("\n\n## ")
        elif tag in ("p", "div", "li", "tr", "br", "figcaption", "blockquote"):
            self.out.append("\n")
        elif tag == "math":
            if a.get("alttext"):
                self.out.append(f" {a['alttext']} ")
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in self.SKIP or tag == "math":
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if not self.skip:
            self.out.append(data)


def html_to_text(page: str) -> tuple[str, dict, list[str], str]:
    """(text, meta tags, links, <title>) of an HTML page."""
    p = _Text()
    p.feed(page)
    text = re.sub(r"[ \t]+", " ", html.unescape("".join(p.out)))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip(), p.meta, list(dict.fromkeys(p.links)), html.unescape(p.title).strip()


def normalize(s: str) -> str:
    """For quote checks: whitespace collapsed, curly quotes straightened, lower case."""
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def section(text: str, names: tuple[str, ...]) -> tuple[str, str]:
    """(heading, text) of the first '## ' section whose heading starts with one of `names`."""
    parts = re.split(r"\n## ", "\n" + text)
    for part in parts[1:]:
        head, _, rest = part.partition("\n")
        if any(head.strip().lower().startswith(n.lower()) for n in names) and rest.strip():
            return head.strip(), rest.split("\n## ")[0].strip()
    return "", ""


def github_repos(urls: list[str]) -> list[str]:
    out = []
    for u in urls:
        m = re.match(r"https?://github\.com/([^/\s#?]+)/([^/\s#?]+)", u)
        if m:
            out.append(f"https://github.com/{m.group(1)}/{m.group(2).removesuffix('.git')}")
    return list(dict.fromkeys(out))
