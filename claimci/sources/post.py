"""Posts given as text: X threads and other pages that cannot be fetched. The whole text is both
where the findings are stated and where any evidence is; the URL is the reference."""
from __future__ import annotations

import re

from .document import Document


def from_text(url: str, text: str) -> Document:
    handle = re.match(r"https?://(?:x|twitter)\.com/([^/]+)/status/", url)
    links = re.findall(r"https?://\S+", text)
    return Document(ref=url, kind="post", title="", authors=[f"@{handle.group(1)}"] if handle else [],
                    claims_text=text, claims_locator="", body=text, links=links)
