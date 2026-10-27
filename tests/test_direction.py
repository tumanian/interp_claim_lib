"""Contrastive directions, templates, matching checks and expansion. No model, no network."""
import yaml

import torch

from claimci.sources import templates
from claimci.sources.expand import scale_block
from claimci.run import interventions, readouts
from claimci.schema.loader import load_repo
from claimci.sources.match import check_params
from claimci.checks import validate

SRC = {"id": "src.chen2025.persona_vectors_identified", "kind": "source", "evidence": "TAS", "status": "proposed",
       "statement": "s", "question": "q.chen2025",
       "provenance": {"source": {"kind": "paper", "ref": "arXiv:2507.21509", "locator": "Abstract"}}}


def write_base(tmp_path, claims):
    (tmp_path / "claims").mkdir(exist_ok=True)
    (tmp_path / "questions").mkdir(exist_ok=True)
    (tmp_path / "claims" / f"{SRC['id']}.yaml").write_text(yaml.safe_dump(SRC))
    return templates.write_claims(claims, tmp_path / "claims", tmp_path / "questions")


def test_every_pack_instantiates_valid_claims(tmp_path):
    for pack in templates.PACKS:
        write_base(tmp_path, templates.contrastive_direction(SRC, "Qwen/Qwen2.5-7B-Instruct", 13, pack))
    repo = load_repo(tmp_path)
    assert validate(repo) == []
    tests = [c for c in repo.claims.values() if c.kind == "test"]
    controls = [c for c in tests if c.control_for]
    assert len(controls) == len(templates.PACKS)          # one random-direction control per pack
    assert all(c.expect is not None for c in tests)        # thresholds fixed at instantiation
    assert all(c.provenance["template"]["name"] == "contrastive_direction" for c in repo.claims.values() if not c.is_source)


def test_paper_layer_maps_to_block_per_repository():
    persona = templates.contrastive_direction(SRC, "m/x", 20, "persona_vectors/evil")
    caa = templates.contrastive_direction(SRC, "m/x", 13, "caa/sycophancy")
    assert persona[0]["code"]["params"]["block"] == 19     # persona_vectors hooks layer_idx = layer - 1
    assert caa[0]["code"]["params"]["block"] == 13         # CAA adds at block `layer`
    assert "layer_idx = layer - 1" in persona[0]["provenance"]["supplemented"]["code.params.block"]["note"]


def test_steering_claims_depend_on_the_direction_and_share_its_data():
    cs = {c["id"].split(".")[-1]: c for c in templates.contrastive_direction(SRC, "m/x", 13, "caa/sycophancy")}
    for k in ("steer_up", "steer_down", "steer_up_random"):
        assert cs[k]["depends_on"] == [cs["direction"]["id"]]
        assert cs[k]["inputs"]["direction"].endswith(".direction.direction")
        assert cs[k]["data"] == cs["direction"]["data"]
    assert cs["steer_up_random"]["code"]["params"]["random_seed"] == 0
    assert cs["steer_down"]["code"]["params"]["coef"] == -cs["steer_up"]["code"]["params"]["coef"]


def test_qwen35_runs_unbatched():
    assert templates.default_opts("Qwen/Qwen3.5-4B")["batch"] == 1
    assert templates.default_opts("Qwen/Qwen2.5-7B-Instruct")["batch"] > 1


def test_scale_block_keeps_relative_depth():
    assert scale_block(13, 32, 32) == 13
    assert scale_block(19, 28, 28) == 19
    assert scale_block(13, 32, 28) == 11
    assert scale_block(19, 28, 18) == 12
    assert scale_block(0, 32, 4) == 0 and scale_block(31, 32, 4) == 3


def test_check_params():
    base = {"template": "contrastive_direction", "hf_model": "Qwen/Qwen2.5-7B-Instruct", "packs": ["caa/sycophancy"],
            "layer": 13, "interventions": []}
    assert check_params(base) is None
    assert check_params({**base, "packs": []}) is not None
    assert check_params({**base, "packs": ["caa/nope"]}) is not None
    assert check_params({**base, "layer": -1}) is not None
    assert check_params({**base, "template": ""}) is None   # no template: a gap, not an error


def test_balanced_metric_ignores_letter_bias():
    items = [{"answer_matching_behavior": "(A)"}, {"answer_matching_behavior": "(A)"}, {"answer_matching_behavior": "(B)"}]
    always_a = [1.0, 1.0, 0.0]                              # P(match) for a model that always answers A
    assert readouts.balanced(items, always_a, lambda it: readouts.letter(it["answer_matching_behavior"])) == 0.5
    assert readouts.balanced(items, [1.0, 1.0, 1.0], lambda it: readouts.letter(it["answer_matching_behavior"])) == 1.0


def test_letter_parsing():
    assert readouts.letter("(A)") == "A" and readouts.letter(" (B) ") == "B" and readouts.letter("A") == "A"


class _Block(torch.nn.Module):
    def forward(self, x):
        return (x * 1.0, "extra")


class _H:
    def __init__(self):
        self.layers = [_Block()]


def test_add_hook_all_and_last_positions():
    from claimci.run.backends import registered
    h, v = _H(), torch.tensor([1.0, 2.0])
    x = torch.zeros(1, 3, 2)
    with registered(interventions.add_vector(h, 0, v, 2.0, "all")):
        out = h.layers[0](x)[0]
    assert torch.equal(out, torch.tensor([[[2.0, 4.0]] * 3]))
    with registered(interventions.add_vector(h, 0, v, 2.0, "last")):
        out = h.layers[0](x)[0]
    assert torch.equal(out[0, :2], torch.zeros(2, 2)) and torch.equal(out[0, 2], torch.tensor([2.0, 4.0]))
    assert torch.equal(h.layers[0](x)[0], x)               # hooks removed on exit
