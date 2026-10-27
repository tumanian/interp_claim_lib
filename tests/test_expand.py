"""expand: clones a finding's claims onto another model by declared parameter kinds. No network:
layer counts are given by the test."""
import yaml

from claimci import expand as E
from claimci import templates
from claimci.loader import load_repo
from claimci.validate import validate

SRC = {"id": "src.p.finding", "kind": "source", "evidence": "TAS", "status": "proposed", "statement": "s",
       "question": "q.p", "provenance": {"source": {"kind": "paper", "ref": "arXiv:1", "locator": "Abstract"}}}
LAYERS = {"meta-llama/Llama-2-7b-chat-hf": 32, "Qwen/Qwen2.5-7B-Instruct": 28, "Qwen/Qwen3.5-4B": 32}


def base(tmp_path, claims, monkeypatch):
    (tmp_path / "claims").mkdir(); (tmp_path / "questions").mkdir()
    (tmp_path / "claims" / "src.p.finding.yaml").write_text(yaml.safe_dump(SRC))
    (tmp_path / "questions" / "q.p.yaml").write_text(yaml.safe_dump({"id": "q.p", "text": "t", "claims": []}))
    templates.write_claims(claims, tmp_path / "claims", tmp_path / "questions")
    monkeypatch.setattr(E, "n_layers", lambda m: LAYERS[m])
    return tmp_path


def test_caa_closure_moves_with_layers_scaled_and_links_rewired(tmp_path, monkeypatch):
    root = base(tmp_path, templates.contrastive_direction(SRC, "meta-llama/Llama-2-7b-chat-hf", 13, "caa/sycophancy"), monkeypatch)
    out = E.expand(root, "arXiv:1", ["Qwen/Qwen2.5-7B-Instruct"], log=lambda s: None)
    assert out["errors"] == [] and out["refused"] == [] and len(out["written"]) == 4
    repo = load_repo(root)
    new = {k.split(".")[-1]: repo.claims[k] for k in out["written"]}
    assert all(c.model == "Qwen/Qwen2.5-7B-Instruct" and c.provenance["expanded_from"] == "meta-llama/Llama-2-7b-chat-hf"
               for c in new.values())
    assert new["direction"].params["block"] == 11                      # 13 of 32 -> 11 of 28
    assert "relative depth" in new["direction"].provenance["supplemented"]["code.params.block"]["note"]
    assert new["steer_up"].depends_on == [new["direction"].id]          # rewired to the clone, not the original
    assert new["steer_up"].inputs["direction"] == f"{new['direction'].id}.direction"
    assert new["steer_up_random"].control_for == new["steer_up"].id
    assert new["steer_up"].experiment_id == "caa.steer" and new["steer_up"].about == "src.p.finding"
    assert validate(repo) == []


def test_neuron_indices_refuse_the_closure_with_a_reason(tmp_path, monkeypatch):
    claims = templates.neuron_steering(SRC, "Qwen/Qwen3.5-4B", [{"layer": 14, "neuron": 8949, "dose": 1}])
    root = base(tmp_path, claims, monkeypatch)
    out = E.expand(root, "arXiv:1", ["Qwen/Qwen2.5-7B-Instruct"], log=lambda s: None)
    assert out["written"] == [] and out["refused"]
    assert all("component parameter" in r or "interventions" in r for r in out["refused"])


def test_expansions_are_never_expanded_again_and_never_overwritten(tmp_path, monkeypatch):
    root = base(tmp_path, templates.contrastive_direction(SRC, "meta-llama/Llama-2-7b-chat-hf", 13, "caa/sycophancy"), monkeypatch)
    first = E.expand(root, "arXiv:1", ["Qwen/Qwen2.5-7B-Instruct"], log=lambda s: None)
    again = E.expand(root, "arXiv:1", ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.5-4B"], log=lambda s: None)
    assert len(first["written"]) == 4 and len(again["written"]) == 4   # only the new model; nothing cloned from a clone
    repo = load_repo(root)
    assert all(repo.claims[k].provenance["expanded_from"] == "meta-llama/Llama-2-7b-chat-hf" for k in again["written"])


def test_scale_block():
    assert E.scale_block(13, 32, 28) == 11 and E.scale_block(19, 28, 18) == 12 and E.scale_block(0, 32, 4) == 0
