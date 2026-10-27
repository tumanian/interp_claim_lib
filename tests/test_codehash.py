"""Behaviour hash and registry: independent of layout, sensitive to behaviour. Works on a copy of
the package, so edits here never touch the real source."""
import shutil
from pathlib import Path

import pytest

from claimci import codehash

PKG = Path(codehash.__file__).resolve().parent


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    dst = tmp_path / "claimci"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(codehash, "PKG_DIR", dst)
    codehash.clear_cache()
    yield dst
    codehash.clear_cache()


def h(eid):
    codehash.clear_cache()
    r = codehash.registry()[eid]
    return codehash.behaviour_hash(r.module, r.function)


def edit(path, old, new):
    s = path.read_text()
    assert old in s, old
    path.write_text(s.replace(old, new, 1))


def test_registry_is_found_without_importing(pkg):
    r = codehash.registry()
    assert r["caa.steer"].entrypoint == "claimci.experiments.direction:steer_ab"
    assert "intervention" in r["caa.steer"].tags and "intervention" not in r["caa.direction"].tags
    assert dict(r["caa.steer"].params)["block"] == "layer"
    assert dict(r["steering.persona_delta"].params)["interventions"] == "component"


def test_comments_docstrings_imports_and_decorators_do_not_count(pkg):
    before = h("caa.steer")
    f = pkg / "experiments" / "direction.py"
    edit(f, '"""Balanced P(behaviour-matching answer) on the test split, steered minus unsteered."""', '"""Reworded."""')
    edit(f, "import torch\n", "import torch  # a comment\nimport os\n")
    edit(f, "    items = _pack(acts, pack)[\"test\"]\n", "    # a comment inside\n    items = _pack(acts, pack)[\"test\"]\n")
    assert h("caa.steer") == before


def test_moving_a_function_to_another_module_does_not_count(pkg):
    before = h("persona.steer")
    src = pkg / "experiments" / "direction.py"
    s = src.read_text()
    start = s.index("def _trait_scores(")
    end = s.index("\n\n\n", start) + 3
    helper = s[start:end]
    src.write_text(s[:start] + s[end:])            # move the helper to a new module, imported from there
    (pkg / "experiments" / "persona_helpers.py").write_text(
        "from .. import operations as ops\nfrom ..formats import format_prompt\n\n\ndef _judge(pairs, prompts, model, effort):\n"
        "    from .judge import judge\n    return judge(pairs, {\"aligned\": prompts[\"trait\"], \"coherent\": prompts[\"coherence\"]}, model, effort)\n\n\n"
        + helper)
    edit(src, "from .. import operations as ops\n", "from .. import operations as ops\nfrom .persona_helpers import _trait_scores\n")
    assert h("persona.steer") == before


def test_changing_a_helper_in_another_module_counts(pkg):
    before = h("caa.steer")
    edit(pkg / "operations.py", "lp = F.log_softmax(h.model(**enc).logits[:, -1, :].to(torch.float64), dim=-1).cpu()",
         "lp = F.log_softmax(h.model(**enc).logits[:, -1, :].to(torch.float32), dim=-1).cpu()")
    assert h("caa.steer") != before                # operations.last_logprobs, via readouts.ab_matching


def test_changing_a_constant_counts(pkg):
    judge_before, persona_before = h("judge.misalignment"), h("steering.persona_delta")
    edit(pkg / "experiments" / "judge.py", "MISALIGNED_BELOW, COHERENT_ABOVE = 30, 50", "MISALIGNED_BELOW, COHERENT_ABOVE = 25, 50")
    edit(pkg / "experiments" / "steering.py", 'QUESTION_SUFFIX = "\\n\\nAnswer with Yes or No."', 'QUESTION_SUFFIX = "\\n\\nYes or No?"')
    assert h("judge.misalignment") != judge_before          # a threshold in a tuple assignment
    assert h("steering.persona_delta") != persona_before    # a prompt suffix


def test_unrelated_changes_elsewhere_do_not_count(pkg):
    before = h("caa.steer")
    edit(pkg / "experiments" / "steering.py", "def _by_match(it):", "def _by_match(it, unused=None):")
    assert h("caa.steer") == before


def test_duplicate_ids_are_rejected(pkg):
    edit(pkg / "experiments" / "direction.py", '@experiment("caa.direction"', '@experiment("caa.steer"')
    codehash.clear_cache()
    with pytest.raises(ValueError, match="registered twice"):
        codehash.registry()
