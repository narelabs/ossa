"""Correctness tests for the true sparse attention forward."""

import torch

from ossa.sparse_attention import (
    dense_attention_reference,
    sparse_attention_forward,
    sparse_attention_forward_chunked,
)


def test_full_topk_matches_dense():
    """If we pass topk_indices = all positions, sparse must equal dense."""
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 32, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # All N indices for every query
    topk = torch.arange(N).view(1, 1, 1, N).expand(B, H, N, N).contiguous()
    out_sparse = sparse_attention_forward(q, k, v, topk_indices=topk)
    out_dense = dense_attention_reference(q, k, v)

    assert torch.allclose(out_sparse, out_dense, atol=1e-5), \
        f"max diff = {(out_sparse - out_dense).abs().max().item()}"


def test_chunked_matches_full():
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 64, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    K = 16

    # Pick top-K of dense scores per query as the "router output"
    scores = (q @ k.transpose(-2, -1))
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    topk = scores.topk(K, dim=-1).indices

    out_full = sparse_attention_forward(q, k, v, topk_indices=topk)
    out_chunked = sparse_attention_forward_chunked(q, k, v, topk_indices=topk, chunk_size=16)

    assert torch.allclose(out_full, out_chunked, atol=1e-5)


def test_oracle_topk_close_to_dense():
    """Top-K of dense scores should reproduce dense output very well
    when the attention is concentrated."""
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 64, 16
    q = torch.randn(B, H, N, D) * 3   # high temperature → concentrated
    k = torch.randn(B, H, N, D) * 3
    v = torch.randn(B, H, N, D)
    K = 8

    scores = (q @ k.transpose(-2, -1))
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    topk = scores.topk(K, dim=-1).indices

    out_sparse = sparse_attention_forward(q, k, v, topk_indices=topk)
    out_dense = dense_attention_reference(q, k, v)

    # cosine similarity between the two outputs should be very high
    cos = torch.nn.functional.cosine_similarity(
        out_sparse.flatten(end_dim=-2), out_dense.flatten(end_dim=-2), dim=-1
    ).mean().item()
    assert cos > 0.95, f"cosine = {cos}"
