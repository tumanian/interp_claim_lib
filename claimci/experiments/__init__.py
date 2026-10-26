"""Local experiment library."""


def intervention(fn):
    """Tag an entrypoint as an intervention (needs `ctx` to rerun the model with hooks).

    The validator detects this decorator statically; claims with causal evidence
    (IMP, ISS, ICA) must use a tagged entrypoint.
    """
    fn.intervention = True
    return fn
