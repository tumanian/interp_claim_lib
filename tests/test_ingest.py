"""Ingest's mechanical checks and claim writing. No network, no API calls."""
import yaml

from claimci import ingest
from claimci.loader import load_repo
from claimci.validate import validate

ABSTRACT = "We find that “refusal is mediated by a one-dimensional subspace” across 13 models."
BODY = """## 3.1 Bypassing refusal
Ablating the direction reduces the refusal rate from 0.91 to 0.00 on JailbreakBench."""
TEST = {"description": "ablate", "locator": "Section 3.1", "evidence": "ablation_or_patching", "models": ["m"],
        "data": "JailbreakBench", "metric": "refusal rate", "quote": "reduces the refusal  rate from 0.91",
        "stated_values": [{"name": "refusal before", "value": "0.91", "context": "c"},
                          {"name": "invented", "value": "0.42", "context": "c"}]}

OUT = {"title": "T", "authors": ["Andy Arditi"], "date": "2024-06-17", "references": {"code": [], "data": [], "models": []},
       "claims": [
           {"slug": "ablation_bypasses", "statement": "s", "quote": 'refusal is mediated by a one-dimensional  subspace',
            "hedge": "shows", "scope": "x", "tests": [TEST]},
           {"slug": "made_up", "statement": "s", "quote": "a sentence that is not in the text",
            "hedge": "suggests", "scope": "x", "tests": []}]}


def test_checks_find_real_quotes_and_numbers_only():
    a, b = (ingest.check(ABSTRACT, BODY, c) for c in OUT["claims"])
    assert a["quote_found"] and a["tests"][0]["quote_found"]
    assert a["tests"][0]["values_found"] == {"0.91": True, "0.42": False}
    assert not b["quote_found"] and b["tests"] == []
    assert not ingest.check(BODY, ABSTRACT, OUT["claims"][0])["quote_found"]   # claim quotes come from the abstract


def test_html_to_text_keeps_headings_and_math_alt():
    p = ingest._Text()
    p.feed('<nav>skip</nav><h2>3 Method</h2><p>We add <math alttext="r_l"><mi>r</mi></math> here.</p>')
    text = "".join(p.out)
    assert "## 3 Method" in text and "r_l" in text and "skip" not in text


def test_written_claims_validate_and_never_overwrite(tmp_path):
    (tmp_path / "claims").mkdir()
    (tmp_path / "questions").mkdir()
    claims = ingest.to_claims(OUT, ABSTRACT, BODY, "paper", "arXiv:2406.11717",
                              ingest.prefix_for(OUT, "arXiv:2406.11717"), None)
    assert [c["id"] for c in claims] == ["src.arditi2024.ablation_bypasses", "src.arditi2024.made_up"]
    assert claims[0]["provenance"]["source"]["locator"] == "Abstract" and claims[0]["evidence"] == "TAS"
    assert claims[0]["provenance"]["source"]["stated_evidence"] == "IMP"
    assert claims[0]["stated"]["tests"][0]["locator"] == "Section 3.1" and "stated" not in claims[1]
    written, skipped = ingest.write(claims, tmp_path / "claims")
    assert len(written) == 2 and skipped == []
    assert validate(load_repo(tmp_path)) == []
    again, skipped = ingest.write(claims, tmp_path / "claims")
    assert again == [] and len(skipped) == 2
    d = yaml.safe_load((tmp_path / "claims" / "src.arditi2024.made_up.yaml").read_text())
    assert d["provenance"]["checks"]["quote_found"] is False and d["provenance"]["reviewed"] is False
