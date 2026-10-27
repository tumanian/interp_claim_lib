"""Vendor the small, fixed datasets the steering and refusal claims run on.

Writes data/*.json once; the claim files declare each file's content hash, so a
re-vendor that changes anything shows up in `claimci validate`. Needs network.

  refusal_direction/gemma_pipeline_prompts.json
      The prompt samples of Arditi et al.'s pipeline (github.com/andyrdt/refusal_direction
      at a pinned commit), replayed with its own sampling: random.seed(42), then
      harmful/harmless train (128 each), harmful/harmless val (32 each), and later
      harmless test (100). The harmless test sample is checked against the
      instructions in the authors' published gemma-2b-it completions.
  refusal_direction/gemma-2b-it_direction.pt
      The authors' published direction for gemma-2b-it (layer 10, position -2).
  persona/evil_persona_200.json
      50 items each from four Anthropic model-written-evals persona sets
      (psychopathy, machiavellianism, narcissism, ends-justify-means), seed 0.
  natural_text/wikitext2_64.json
      The first 64 wikitext-2 test paragraphs of at least 300 characters.
"""
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
RAW = f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{COMMIT}"
PERSONA = ["psychopathy", "machiavellianism", "narcissism", "ends-justify-means"]


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT.parent)}  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


def refusal() -> None:
    def split(harmtype, name):
        return [d["instruction"] for d in json.loads(get(f"{RAW}/dataset/splits/{harmtype}_{name}.json"))]

    random.seed(42)  # the pipeline's order of draws, run_pipeline.py
    out = {"source": {"repo": "https://github.com/andyrdt/refusal_direction", "commit": COMMIT}}
    out["harmful_train"] = random.sample(split("harmful", "train"), 128)
    out["harmless_train"] = random.sample(split("harmless", "train"), 128)
    out["harmful_val"] = random.sample(split("harmful", "val"), 32)
    out["harmless_val"] = random.sample(split("harmless", "val"), 32)
    out["harmless_test"] = random.sample(split("harmless", "test"), 100)
    out["jailbreakbench"] = [d["instruction"] for d in json.loads(get(f"{RAW}/dataset/processed/jailbreakbench.json"))]
    published = json.loads(get(f"{RAW}/pipeline/runs/gemma-2b-it/completions/harmless_baseline_completions.json"))
    published = [d.get("prompt") or d.get("instruction") for d in published]
    if published != out["harmless_test"]:
        sys.exit("harmless_test sample does not match the published gemma-2b-it completions; not writing")
    print("harmless_test sample matches the authors' published completions (100/100, same order)")
    write_json(ROOT / "refusal_direction" / "gemma_pipeline_prompts.json", out)
    pt = ROOT / "refusal_direction" / "gemma-2b-it_direction.pt"
    pt.write_bytes(get(f"{RAW}/pipeline/runs/gemma-2b-it/direction.pt"))
    print(f"wrote {pt.relative_to(ROOT.parent)}  sha256 {hashlib.sha256(pt.read_bytes()).hexdigest()}")


def persona() -> None:
    rng, items = random.Random(0), []
    for name in PERSONA:
        url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/persona/{name}.jsonl"
        rows = [json.loads(line) for line in get(url).decode().splitlines() if line.strip()]
        for r in rng.sample(rows, 50):
            items.append({"set": name, "statement": r["statement"], "question": r["question"],
                          "matching": r["answer_matching_behavior"].strip()})
    write_json(ROOT / "persona" / "evil_persona_200.json",
               {"source": "https://huggingface.co/datasets/Anthropic/model-written-evals (CC-BY-4.0)",
                "items": items})


def natural_text() -> None:
    paras, offset = [], 0
    while len(paras) < 64:
        url = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext"
               f"&config=wikitext-2-raw-v1&split=test&offset={offset}&length=100")
        rows = json.loads(get(url))["rows"]
        if not rows:
            break
        paras += [r["row"]["text"].strip() for r in rows if len(r["row"]["text"].strip()) >= 300]
        offset += 100
    write_json(ROOT / "natural_text" / "wikitext2_64.json",
               {"source": "https://huggingface.co/datasets/Salesforce/wikitext wikitext-2-raw-v1 test (CC-BY-SA)",
                "texts": paras[:64]})


if __name__ == "__main__":
    refusal()
    persona()
    natural_text()
