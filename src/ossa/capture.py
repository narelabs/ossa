"""Capture attention scores from every layer of a frozen HF causal LM.

This is the data plane for the OSSA project: we run a single forward pass
on a long input, intercept every self-attention block, and write its
``(num_heads, seq_len, seq_len)`` attention probability matrix to a list.

We use ``output_attentions=True`` together with ``attn_implementation="eager"``
so transformers gives us the post-softmax attention matrix on the public
``outputs.attentions`` tuple. Flash and SDPA implementations skip
materialising that matrix, which is exactly the cost we are trying to avoid
at *inference* time — but we need it once during *capture* to learn what
the router should imitate.
"""

from __future__ import annotations

from typing import Any

import torch


def load_model(name: str, *, device: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def capture_attention_scores(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    seq_len: int,
) -> dict:
    """Return ``{"layers": [(B,H,N,N), ...], "ids": (1,N)}``."""

    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    out = model(
        input_ids=ids,
        attention_mask=attn_mask,
        output_attentions=True,
        use_cache=False,
    )

    return {
        "layers": [a.detach().cpu().float() for a in out.attentions],
        "ids": ids.detach().cpu(),
    }
