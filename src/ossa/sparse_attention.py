"""True sparse attention forward.

The patched forward in ``bench/sparse_forward.py`` was a *correctness*
baseline: it computes the full ``Q·K^T`` (quadratic), then masks
non-top-K positions, then softmax. Same number of FLOPs as dense
attention. Useful to measure perplexity but says nothing about speed.

This module implements the **real** sparse attention: for each query,
gather only the K selected keys/values, compute K dot products,
softmax over K, and weighted-sum K values. Per query the cost is
O(K) memory accesses and O(K · d) flops, not O(N · d). Total cost
O(N · K · d) instead of O(N² · d). On long sequences this is the
substantive speedup of any sparse-attention method.

We use ``torch.gather`` so it runs in plain PyTorch — slower than a
hand-tuned Triton kernel but algorithmically correct and useful for
wall-clock benchmarking against dense.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    topk_indices: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Compute attention with only ``topk_indices`` keys per query.

    Parameters
    ----------
    q : (B, H, N, D)         queries
    k : (B, H, N, D)         keys
    v : (B, H, N, D)         values
    topk_indices : (B, H, N, K)  indices into ``N`` per query.
        Must already be causal: the caller is responsible for not
        including positions ``> i`` for query ``i``.
    causal : bool
        If True, additionally mask any indices > query position.
        Cheap insurance; pass False if you've already filtered.

    Returns
    -------
    out : (B, H, N, D)
    """
    B, H, N, D = q.shape
    K = topk_indices.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # gather K keys per (B, H, N): (B, H, N, K, D)
    idx_expand = topk_indices.unsqueeze(-1).expand(B, H, N, K, D)
    k_full = k.unsqueeze(2).expand(B, H, N, N, D)
    k_topk = torch.gather(k_full, dim=3, index=idx_expand)  # (B, H, N, K, D)

    v_full = v.unsqueeze(2).expand(B, H, N, N, D)
    v_topk = torch.gather(v_full, dim=3, index=idx_expand)  # (B, H, N, K, D)

    # q (B, H, N, D) → (B, H, N, 1, D); dot with k_topk (B, H, N, K, D)
    scores = (q.unsqueeze(-2) * k_topk).sum(dim=-1) * scale  # (B, H, N, K)

    if causal:
        # mask any topk index > query position
        positions_q = torch.arange(N, device=q.device).view(1, 1, N, 1)
        out_of_causal = topk_indices > positions_q
        scores = scores.masked_fill(out_of_causal, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # rows where all -inf
    out = (attn.unsqueeze(-1) * v_topk).sum(dim=-2)  # (B, H, N, D)
    return out


# ---------------------------------------------------------------------------
# Memory-frugal variant
# ---------------------------------------------------------------------------


def sparse_attention_forward_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    topk_indices: torch.Tensor,
    chunk_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    """True O(N·K) sparse forward without the N×N expand.

    For each chunk of C query rows, we directly index K and V using
    ``topk_indices`` flattened to a 2D batch index. No (..., N, ...)
    intermediate is materialised.
    """
    B, H, N, D = q.shape
    K = topk_indices.shape[-1]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)

    # Flatten batch×heads so we can use simple advanced indexing
    BH = B * H
    k_flat = k.reshape(BH, N, D)             # (BH, N, D)
    v_flat = v.reshape(BH, N, D)
    bh_idx = torch.arange(BH, device=q.device).view(BH, 1, 1)  # (BH, 1, 1)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        C = end - start
        q_c = q[:, :, start:end, :].reshape(BH, C, D)         # (BH, C, D)
        idx_c = topk_indices[:, :, start:end, :].reshape(BH, C, K)  # (BH, C, K)

        # Advanced indexing: k_flat[bh_idx (BH,1,1), idx_c (BH,C,K)]
        # → result shape (BH, C, K, D), allocates only C·K·D per chunk.
        bh_b = bh_idx.expand(BH, C, K)
        k_topk = k_flat[bh_b, idx_c]                          # (BH, C, K, D)
        v_topk = v_flat[bh_b, idx_c]                          # (BH, C, K, D)

        # scores: (BH, C, K)
        scores = torch.einsum("bcd,bckd->bck", q_c, k_topk) * scale

        if causal:
            positions_q = torch.arange(start, end, device=q.device).view(1, C, 1)
            scores = scores.masked_fill(idx_c > positions_q, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out_c = torch.einsum("bck,bckd->bcd", attn, v_topk)   # (BH, C, D)
        out[:, :, start:end, :] = out_c.reshape(B, H, C, D)

    return out


def dense_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True
) -> torch.Tensor:
    """Reference dense attention for tests."""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
