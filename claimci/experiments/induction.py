"""Induction-head experiments. Implemented in step 4; signatures fixed here."""
from . import intervention


def prev_token_score(acts):
    raise NotImplementedError


def induction_head_score(acts, min_layer):
    raise NotImplementedError


@intervention
def ablate_head_loss_delta(acts, ctx, layer, head, ablation="zero"):
    raise NotImplementedError
