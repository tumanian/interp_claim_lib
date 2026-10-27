"""Sources and extraction without network or API: adapters on fixtures, checks, the stored shape."""
import yaml

from claimci import source_tests
from claimci.loader import load_repo
from claimci.sources import Document, route
from claimci.sources import arxiv, extract, web
from claimci.sources.document import html_to_text, section
from claimci.sources.post import from_text
from claimci.validate import validate

DOC = Document(ref="arXiv:2406.11717", kind="paper", title="T", authors=["Andy Arditi"], date="2024-06-17",
               claims_text="We find that “refusal is mediated by a one-dimensional subspace” across 13 models.",
               claims_locator="Abstract",
               body="## 3.1 Bypassing refusal\nAblating the direction reduces the refusal rate from 0.91 to 0.00 on JailbreakBench.")
TEST = {"description": "ablate", "locator": "Section 3.1", "evidence": "ablation_or_patching", "models": ["m"],
        "data": "JailbreakBench", "metric": "refusal rate", "quote": "reduces the refusal  rate from 0.91",
        "stated_values": [{"name": "refusal before", "value": "0.91", "context": "c"},
                          {"name": "invented", "value": "0.42", "context": "c"}]}
OUT = {"title": "T", "authors": ["Andy Arditi"], "date": "2024-06-17",
       "references": {"own_code": ["https://github.com/andyrdt/refusal_direction"], "code": [], "data": [], "models": []},
       "claims": [
           {"slug": "ablation_bypasses", "statement": "s", "quote": "refusal is mediated by a one-dimensional  subspace",
            "hedge": "shows", "scope": "x", "tests": [TEST]},
           {"slug": "made_up", "statement": "s", "quote": "a sentence that is not in the text",
            "hedge": "suggests", "scope": "x", "tests": []}]}


def test_checks_find_real_quotes_and_numbers_only():
    a, b = (extract.check(DOC, c) for c in OUT["claims"])
    assert a["quote_found"] and a["tests"][0]["quote_found"]
    assert a["tests"][0]["values_found"] == {"0.91": True, "0.42": False}
    assert not b["quote_found"] and b["tests"] == []
    swapped = Document(ref="x", kind="paper", claims_text=DOC.body, body=DOC.claims_text)
    assert not extract.check(swapped, OUT["claims"][0])["quote_found"]   # quotes are checked where they came from


def test_written_claims_validate_and_never_overwrite(tmp_path):
    (tmp_path / "claims").mkdir()
    (tmp_path / "questions").mkdir()
    claims = extract.to_claims(OUT, DOC, extract.prefix_for(OUT, DOC.ref), None)
    assert [c["id"] for c in claims] == ["src.arditi2024.ablation_bypasses", "src.arditi2024.made_up"]
    assert claims[0]["evidence"] == "TAS" and claims[0]["provenance"]["source"]["locator"] == "Abstract"
    assert claims[0]["provenance"]["source"]["stated_evidence"] == "IMP"
    assert claims[0]["provenance"]["source"]["code"] == ["https://github.com/andyrdt/refusal_direction"]
    assert claims[0]["stated"]["tests"][0]["locator"] == "Section 3.1" and "stated" not in claims[1]
    written, skipped = extract.write(claims, tmp_path / "claims")
    assert len(written) == 2 and skipped == []
    assert validate(load_repo(tmp_path)) == []
    again, skipped = extract.write(claims, tmp_path / "claims")
    assert again == [] and len(skipped) == 2
    d = yaml.safe_load((tmp_path / "claims" / "src.arditi2024.made_up.yaml").read_text())
    assert d["provenance"]["checks"]["quote_found"] is False and d["provenance"]["reviewed"] is False


def test_stored_tests_are_validated_by_their_own_type(tmp_path):
    assert source_tests.errors([TEST]) == []
    assert source_tests.errors([{**TEST, "evidence": "vibes"}])
    assert source_tests.errors([{**TEST, "surprise": 1}])
    assert source_tests.errors([{**TEST, "stated_values": [{"name": "x", "value": 0.5}]}])   # values stay text
    assert source_tests.errors({"not": "a list"})
    (tmp_path / "claims").mkdir(); (tmp_path / "questions").mkdir()
    bad = extract.to_claims(OUT, DOC, "p2024", None)[0]
    bad["stated"]["tests"][0]["evidence"] = "vibes"
    (tmp_path / "claims" / f"{bad['id']}.yaml").write_text(yaml.safe_dump(bad))
    assert any("stated.tests[0].evidence" in e for e in validate(load_repo(tmp_path)))


def test_html_to_text_keeps_headings_math_meta_and_links():
    text, meta, links, title = html_to_text(
        '<html><head><title>A page</title><meta property="og:title" content="Better title">'
        '<meta name="author" content="Ada"></head><body><nav>skip</nav><h2>3 Method</h2>'
        '<p>We add <math alttext="r_l"><mi>r</mi></math> here, <a href="https://github.com/a/b">code</a>.</p></body></html>')
    assert "## 3 Method" in text and "r_l" in text and "skip" not in text
    assert meta["og:title"] == "Better title" and meta["author"] == "Ada" and title == "A page"
    assert links == ["https://github.com/a/b"]


def test_web_adapter_takes_findings_from_the_summary_when_there_is_one():
    page = ('<html><head><meta property="og:title" content="Steering vectors in practice">'
            '<meta property="article:published_time" content="2025-03-04T10:00:00Z"><meta name="author" content="B. Writer"></head>'
            '<body><h1>Steering vectors in practice</h1><p>Some preamble.</p><h2>TL;DR</h2><p>Adding the vector makes the model refuse.</p>'
            '<h2>Method</h2><p>We added it at layer 10 and refusal went from 0.1 to 0.9.</p></body></html>')
    doc = web.from_html("https://example.org/post", page)
    assert doc.kind == "blog" and doc.title == "Steering vectors in practice" and doc.date == "2025-03-04"
    assert doc.authors == ["B. Writer"] and doc.claims_locator == "TL;DR"
    assert doc.claims_text == "Adding the vector makes the model refuse."
    assert "from 0.1 to 0.9" in doc.body
    no_summary = web.from_html("https://example.org/p", "<html><body><p>Just text about heads.</p></body></html>")
    assert no_summary.claims_locator == "" and "Just text" in no_summary.claims_text


def test_post_adapter_and_routing(monkeypatch):
    doc = from_text("https://x.com/someone/status/1", "steer L14 N8949 and see https://example.org/demo")
    assert doc.kind == "post" and doc.authors == ["@someone"] and doc.claims_text == doc.body
    assert doc.links == ["https://example.org/demo"] and doc.claims_locator == ""
    assert route("https://x.com/a/status/2", text="hi").kind == "post"
    monkeypatch.setattr(arxiv, "fetch", lambda s: Document(ref="arXiv:1", kind="paper"))
    monkeypatch.setattr(web, "fetch", lambda s: Document(ref=s, kind="blog"))
    assert route("2406.11717").kind == "paper" and route("arXiv:2406.11717v2").kind == "paper"
    assert route("https://arxiv.org/abs/2406.11717").kind == "paper"
    assert route("https://transformer-circuits.pub/2023/x").kind == "blog"


def test_arxiv_metadata_comes_from_the_api(monkeypatch):
    xml = ("<feed><entry><title>Refusal  Is\n Mediated</title><summary> Conversational models ...</summary>"
           "<published>2024-06-17T17:00:00Z</published><author><name>Andy Arditi</name></author>"
           "<author><name>Neel Nanda</name></author></entry></feed>")
    monkeypatch.setattr(arxiv, "_get", lambda url: xml if "export.arxiv.org" in url else "<html><body><p>Body.</p></body></html>")
    doc = arxiv.fetch("arXiv:2406.11717v3")
    assert doc.ref == "arXiv:2406.11717" and doc.title == "Refusal Is Mediated" and doc.date == "2024-06-17"
    assert doc.authors == ["Andy Arditi", "Neel Nanda"] and doc.claims_text == "Conversational models ..."
    assert doc.claims_locator == "Abstract" and "Body." in doc.body


def test_section_finds_the_first_matching_heading():
    head, body = section("intro\n\n## Summary\nshort\n\n## Results\nlong", ("TL;DR", "Summary"))
    assert head == "Summary" and body == "short"
