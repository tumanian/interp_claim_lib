"""Formats: how an example becomes text for a model, independent of what is then done with it.

Chat is a format (system / user / assistant turns, the model's template, an assistant prefix);
inference and training are operations that can consume it (claimci/operations.py).
"""
from __future__ import annotations

GEMMA_PAPER_TEMPLATE = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def format_prompt(h, instruction: str, system: str | None = None, prefix: str = "") -> str:
    """The chat-formatted prompt, ready for the assistant's first token. `system` adds a system turn
    (folded into the user turn for templates without one); `prefix` starts the assistant's answer."""
    template = h.opts.get("template", "hf")
    if template == "gemma_paper":
        text = f"{system}\n\n{instruction}" if system else instruction
        return GEMMA_PAPER_TEMPLATE.format(instruction=text) + prefix
    kw = {"enable_thinking": False} if h.opts.get("thinking") is False else {}
    msgs = [{"role": "user", "content": instruction}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    try:
        out = h.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # no system role in this template: fold it into the user turn
        msgs = [{"role": "user", "content": f"{system}\n\n{instruction}"}]
        out = h.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return out + prefix


def with_bos(h, texts: list[str]) -> list[str]:
    """Prepend BOS where the tokenizer has one and the text does not start with it."""
    b = h.tokenizer.bos_token
    return [t if not b or t.startswith(b) else b + t for t in texts]
