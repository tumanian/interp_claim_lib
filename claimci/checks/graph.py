"""Claims as a graph: cycles, and references to other claims, questions and exports."""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from .rules import Rule, broken, first_broken


def cycles(repo) -> list[str]:
    graph = {cid: [d for d in c.depends_on if d in repo.claims] for cid, c in repo.claims.items()}
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as e:
        cyc = list(reversed(e.args[1]))          # graphlib lists it child-first; show it as written
        return [f"claim {cyc[0]}: cycle in depends_on: {' -> '.join(cyc)}"]
    return []


QUESTION = [   # a source claim may have no question; a test or measurement must
    Rule("question {c.question!r} not found",
         when=lambda c, **_: not c.is_source or c.question is not None,
         ok=lambda c, repo, **_: c.question in repo.questions),
]

REFERENCES = QUESTION + [
    Rule("control_for must be on a test and name an existing claim, got {c.control_for!r}",
         when=lambda c, **_: c.control_for is not None,
         ok=lambda c, repo, **_: c.control_for in repo.claims and c.kind == "test"),
    Rule("about must name a source claim, got {c.about!r}",
         when=lambda c, **_: c.about is not None,
         ok=lambda c, repo, **_: c.about in repo.claims and repo.claims[c.about].is_source),
]

INPUT = [   # one named input `name: <claim_id>.<export>`; the first broken rule is reported
    Rule("input {name}={ref!r} must be <claim_id>.<export>", ok=lambda ref, **_: "." in ref),
    Rule("input {name} reads from {pid}, which is not in depends_on", ok=lambda c, pid, **_: pid in c.depends_on),
    Rule("input {name} reads {pid}.{export}, but {pid} exports {parent.exports}",
         when=lambda parent, **_: parent is not None, ok=lambda parent, export, **_: export in parent.exports),
    # Rule 4: exports are consumed on the same data version unless cross_version.
    Rule("input {name} crosses data versions from {pid}; set cross_version: true",
         when=lambda c, parent, **_: parent is not None and not c.cross_version and bool(c.dataset and parent.dataset),
         ok=lambda c, parent, **_: c.dataset.ident == parent.dataset.ident),
]


def references(repo, c) -> list[str]:
    out = broken(REFERENCES, f"claim {c.id}: ", c=c, repo=repo)
    for name, ref in c.inputs.items():
        pid, _, export = ref.rpartition(".")
        out += first_broken(INPUT, f"claim {c.id}: ", c=c, name=name, ref=ref, pid=pid, export=export,
                            parent=repo.claims.get(pid))
    return out


def question_lists(repo) -> list[str]:
    return [f"question {q.id}: lists unknown claim {cid}"
            for q in repo.questions.values() for cid in q.claims if cid not in repo.claims]
