"""Smoke tests for the OSSA sparsity probe helpers."""

from __future__ import annotations

import torch

from ossa.bench.sparsity import _topk_mass


def test_topk_mass_uniform_distribution() -> None:
    # Uniform attention -> top-k mass = k/N
    N = 32
    k_list = [4, 8, 16]
    attn = torch.full((1, 2, N, N), 1.0 / N)  # already normalised rows
    out = _topk_mass(attn, k_list)
    for k in k_list:
        mean, p10, p90 = out[k]
        assert abs(mean - k / N) < 1e-5


def test_topk_mass_one_hot_concentration() -> None:
    # Each query attends to exactly one key with prob 1.0.
    N = 16
    attn = torch.zeros((1, 1, N, N))
    for q in range(N):
        attn[0, 0, q, q] = 1.0
    out = _topk_mass(attn, [1, 4])
    assert out[1][0] == 1.0
    assert out[4][0] == 1.0
