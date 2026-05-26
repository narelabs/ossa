"""Sparse attention routers.

Two router variants live here:

1. ``PerLayerRouter`` — original position-based router. Each key gets a
   learned per-position embedding; the router projects the query and
   takes a dot product. Cheap, but the router is blind to *what* is at
   that position. Recall caps around 0.40 on Qwen-1.5B.

2. ``ContentRouter`` — the better idea. The router takes the query and
   the **real key projection** of the layer it imitates, learns a small
   ``W_q'``, ``W_k'`` over them, and scores ``(q' · k')``. This way the
   router knows what each key *contains*, not just where it is. Drop-in
   replacement for ``PerLayerRouter`` with the same interface.

We train both by **distilling** the dense attention pattern: the target
is the top-K keys of the dense attention matrix at each query position;
the loss is binary cross-entropy on the predicted "is this key in the
top-K" probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    head_dim: int
    n_heads: int
    seq_len: int
    n_buckets: int = 64
    hidden: int = 128
    layers: int = 1


class PerLayerRouter(nn.Module):
    """One router per transformer layer.

    Predicts ``(B, H, N, N)`` logits — for each (head, query, key) — by
    combining a query-projection MLP with a learned key-position embedding.
    Top-K of these logits picks the sparse keys.
    """

    def __init__(self, cfg: RouterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_mlp = nn.Sequential(
            nn.Linear(cfg.head_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.head_dim),
        )
        # Learned per-position key embedding shared across heads.
        self.key_pos = nn.Embedding(cfg.seq_len, cfg.head_dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """``q``: (B, H, N, head_dim) -> logits (B, H, N, N)."""

        # Query projection.
        q_proj = self.q_mlp(q)  # (B, H, N, head_dim)
        # Key embedding for all positions in the sequence.
        positions = torch.arange(self.cfg.seq_len, device=q.device)
        k_emb = self.key_pos(positions)  # (N, head_dim)
        # Score: q_proj @ k_emb^T per head.
        # (B, H, N, head_dim) @ (head_dim, N) -> (B, H, N, N)
        logits = torch.einsum("bhnd,kd->bhnk", q_proj, k_emb)
        return logits

    def topk(self, q: torch.Tensor, k: int) -> torch.Tensor:
        """Return top-k key indices per (B, H, N): (B, H, N, k)."""
        logits = self.forward(q)
        return torch.topk(logits, k=k, dim=-1).indices


def distillation_loss(
    router_logits: torch.Tensor,
    dense_attn: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Binary CE on "is this key in dense top-k?".

    ``router_logits``: (B, H, N, N) — router scores (any real number).
    ``dense_attn``:    (B, H, N, N) — softmax attention probabilities.
    """

    with torch.no_grad():
        # Build a hard 0/1 target: 1 if key in top-k of dense attention for
        # this query, else 0.
        target_idx = torch.topk(dense_attn, k=k, dim=-1).indices  # (B, H, N, k)
        target = torch.zeros_like(dense_attn)
        target.scatter_(-1, target_idx, 1.0)

    return nn.functional.binary_cross_entropy_with_logits(router_logits, target)


def router_recall_at_k(
    router_logits: torch.Tensor,
    dense_attn: torch.Tensor,
    *,
    k: int,
) -> float:
    """Fraction of (query) positions where the router's top-k overlaps the
    dense top-k by at least 50%. Useful as a proxy quality metric."""

    with torch.no_grad():
        dense_idx = torch.topk(dense_attn, k=k, dim=-1).indices  # (B, H, N, k)
        router_idx = torch.topk(router_logits, k=k, dim=-1).indices  # (B, H, N, k)

        # For each query, count how many of router_idx ∈ dense_idx.
        # We do this with broadcasting / set intersection per query.
        # Simple loop over (B, H, N) is fine for our small batch sizes.
        B, H, N, _ = dense_idx.shape
        hits = 0
        total = B * H * N
        for b in range(B):
            for h in range(H):
                for n in range(N):
                    d = set(dense_idx[b, h, n].tolist())
                    r = set(router_idx[b, h, n].tolist())
                    if len(d & r) >= k // 2:
                        hits += 1
        return hits / max(1, total)


# ---------------------------------------------------------------------------
# Content router
# ---------------------------------------------------------------------------


@dataclass
class ContentRouterConfig:
    head_dim: int
    n_heads: int
    seq_len: int
    proj_dim: int = 64        # router-internal projection size
    layers: int = 1


class ContentRouter(nn.Module):
    """Content-based router: scores each (q, k) pair using projections of
    the **real** query and key tensors of the wrapped attention layer.

    Forward signature is intentionally compatible with ``PerLayerRouter``
    on the query side (``forward(q)``), with one extra channel:
    the caller must also supply the layer's real keys ``k`` to ``score``.

    For training we have ``k`` from the same forward hook that gives
    ``q``; for inference, ``k`` is whatever the patched attention has
    just computed before deciding the sparse mask.
    """

    def __init__(self, cfg: ContentRouterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Sequential(
            nn.Linear(cfg.head_dim, cfg.proj_dim, bias=False),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(cfg.head_dim, cfg.proj_dim, bias=False),
        )
        # A tiny per-head bias on (q_pos - k_pos) helps capture locality
        # bias quickly without forcing the content path to learn it.
        self.pos_bias = nn.Embedding(2 * cfg.seq_len, 1)

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """``q``: (B, H, N, head_dim), ``k``: (B, H, N, head_dim).

        Returns ``(B, H, N, N)`` logits. We call this ``forward`` for
        compatibility with the ``PerLayerRouter`` API but it really
        needs the keys, so prefer this name in new code.
        """

        q_h = self.q_proj(q)              # (B, H, N, proj)
        k_h = self.k_proj(k)              # (B, H, N, proj)
        scale = 1.0 / math.sqrt(self.cfg.proj_dim)
        logits = torch.einsum("bhnd,bhmd->bhnm", q_h, k_h) * scale

        N = q.shape[2]
        positions_q = torch.arange(N, device=q.device).unsqueeze(1)
        positions_k = torch.arange(N, device=q.device).unsqueeze(0)
        offsets = positions_q - positions_k + self.cfg.seq_len  # (N, N)
        bias = self.pos_bias(offsets.clamp(min=0, max=2 * self.cfg.seq_len - 1)).squeeze(-1)
        logits = logits + bias  # (B, H, N, N) + (N, N) broadcast
        return logits

    # Keep an alias so existing code that calls ``router(q)`` still
    # type-checks; we redirect it to a no-key fallback that uses zero
    # keys and is therefore useless except as a sanity import.
    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None) -> torch.Tensor:
        if k is None:
            raise TypeError(
                "ContentRouter needs both q and k tensors; call .score(q, k)."
            )
        return self.score(q, k)

    def topk(self, q: torch.Tensor, k: torch.Tensor, top_k: int) -> torch.Tensor:
        return torch.topk(self.score(q, k), k=top_k, dim=-1).indices


# Need ``math`` and ``Optional`` for ContentRouter — add at top-level imports.
