"""Objects: model components with an identity, resolved from what claims already carry.

An object id is `<model>@<revision>/<kind>/<locator>`. Nothing about an object is verified by
itself; it anchors the claims that mention it, so everything checked about one component can
be listed together. Sources, in order:
  - stated interventions in a claim's params (`interventions`: layer, neuron) -> mlp_neuron
  - exported bindings of results: `layer` + `head` -> attn_head (one per revision where found);
    a `direction` artifact (with `layer`, `pos`) -> direction
Controls are skipped: the random components they draw are not what the claim is about.
"""
from __future__ import annotations

from .ledger import Ledger
from .report import computed_status, current_row, revision_order
from .spec import Repo


def objects_of(repo: Repo, ledger: Ledger, cid: str) -> list[dict]:
    c = repo.claims[cid]
    if c.is_source or c.control_for or "random" in c.code.get("entrypoint", ""):
        return []
    out = []
    for iv in c.params.get("interventions") or []:
        L, n = int(iv["layer"]), int(iv["neuron"])
        out.append({"id": f"{c.model}@{c.revision}/mlp_neuron/L{L}/N{n}", "kind": "MLP neuron",
                    "label": f"layer {L}, neuron {n}", "model": c.model, "revision": c.revision})
    revs = sorted({r.revision for r in ledger.history(cid)}, key=revision_order)
    for rev in revs:
        r = current_row(repo, ledger, cid, rev)
        # a binding names a component only where the claim holds (or is a readout): elsewhere it is
        # just the argmax of a score that failed, not a component anyone has shown anything about
        if r is None or computed_status(repo, r, ledger) not in ("verified", "measured"):
            continue
        b = r.bindings  # claims that take layer/head or direction as inputs carry them here too
        if "head" in b and "layer" in b:
            out.append({"id": f"{c.model}@{rev}/attn_head/L{b['layer']}/H{b['head']}", "kind": "attention head",
                        "label": f"layer {b['layer']}, head {b['head']}", "model": c.model, "revision": rev})
        if str(b.get("direction", "")).startswith("artifact:"):
            key = b["direction"].split(":")[1][:10]
            out.append({"id": f"{c.model}@{rev}/direction/{key}", "kind": "residual direction",
                        "label": f"layer {b.get('layer')}, position {b.get('pos')} ({key})", "model": c.model, "revision": rev})
    return out


def objects(repo: Repo, ledger: Ledger, claim_ids) -> list[dict]:
    """Objects touched by these claims, each with the claims that mention it."""
    found: dict[str, dict] = {}
    for cid in sorted(claim_ids):
        for o in objects_of(repo, ledger, cid):
            found.setdefault(o["id"], {**o, "claims": []})["claims"].append(cid)
    return sorted(found.values(), key=lambda o: (o["model"], o["kind"], revision_order(o["revision"]), o["id"]))
