"""Triton kernel for top-K sparse attention forward.    

This implements the same computation as
``ossa.sparse_attention.sparse_attention_forward`` but as a fused Triton
GPU kernel. Each program instance handles one ``(batch, head)`` and a
chunk of ``BLOCK_M`` queries; for every query in the chunk it loads only
the K selected keys/values, computes K dot products, runs an online
softmax, and accumulates the weighted sum of values.

Compute is genuinely O(N · K · D) with no full Q·K^T materialised. On a
modern GPU this is the implementation that turns the algorithmic O(N·K)
saving into wall-clock speedup against dense FlashAttention-style
kernels at the same problem size.

Falls back to the pure-PyTorch ``sparse_attention_forward_chunked`` if
Triton is not available (Windows, CPU-only environments). API is
identical so it can be a drop-in.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    topk_indices: torch.Tensor,
    causal: bool = True,
    block_m: int = 32,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """Top-K sparse attention forward.

    Parameters
    ----------
    q, k, v : ``(B, H, N, D)``
    topk_indices : ``(B, H, N, K)`` indices into ``N``
    causal : keep only indices <= query position
    use_triton : force Triton on/off; ``None`` autodetects

    Returns
    -------
    out : ``(B, H, N, D)``
    """
    use = HAS_TRITON and q.is_cuda if use_triton is None else use_triton
    if not use:
        from ossa.sparse_attention import sparse_attention_forward_chunked
        return sparse_attention_forward_chunked(
            q, k, v, topk_indices=topk_indices, causal=causal,
        )
    return _triton_topk_attention(q, k, v, topk_indices, causal, block_m)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


if HAS_TRITON:

    @triton.jit
    def _topk_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, IDX_ptr, Out_ptr,
        stride_qbh, stride_qb, stride_qn, stride_qd,
        stride_kbh, stride_kb, stride_kn, stride_kd,
        stride_vbh, stride_vb, stride_vn, stride_vd,
        stride_ibh, stride_ib, stride_in, stride_ik,
        stride_obh, stride_ob, stride_on, stride_od,
        N, K_TOP, scale,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """One program: one (batch, head) and BLOCK_M consecutive queries."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < N
        d_offsets = tl.arange(0, D)

        q_base = Q_ptr + pid_bh * stride_qbh
        q_ptrs = q_base + m_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        idx_base = IDX_ptr + pid_bh * stride_ibh + m_offsets[:, None] * stride_in
        k_base = K_ptr + pid_bh * stride_kbh
        v_base = V_ptr + pid_bh * stride_vbh

        for kk in range(0, K_TOP, BLOCK_K):
            k_off = kk + tl.arange(0, BLOCK_K)
            k_mask = k_off < K_TOP

            idx = tl.load(
                idx_base + k_off[None, :] * stride_ik,
                mask=m_mask[:, None] & k_mask[None, :], other=0,
            )

            k_ptrs = (
                k_base
                + idx[:, :, None] * stride_kn
                + d_offsets[None, None, :] * stride_kd
            )
            v_ptrs = (
                v_base
                + idx[:, :, None] * stride_vn
                + d_offsets[None, None, :] * stride_vd
            )
            k_vals = tl.load(
                k_ptrs,
                mask=(m_mask[:, None, None] & k_mask[None, :, None]),
                other=0.0,
            )
            v_vals = tl.load(
                v_ptrs,
                mask=(m_mask[:, None, None] & k_mask[None, :, None]),
                other=0.0,
            )

            s = tl.sum(q[:, None, :] * k_vals, axis=2) * scale

            valid = m_mask[:, None] & k_mask[None, :]
            if IS_CAUSAL:
                valid = valid & (idx <= m_offsets[:, None])
            s = tl.where(valid, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            l_new = alpha * l_i + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_vals, axis=1)
            m_i = m_new
            l_i = l_new

        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out = acc / l_safe[:, None]
        out = tl.where(l_i[:, None] > 0, out, 0.0)

        out_base = Out_ptr + pid_bh * stride_obh
        out_ptrs = out_base + m_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od
        tl.store(out_ptrs, out, mask=m_mask[:, None])


def _triton_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    causal: bool,
    block_m: int,
) -> torch.Tensor:
    """Launch the Triton kernel."""
    assert HAS_TRITON, "triton not installed"
    assert q.is_cuda, "Triton kernel requires CUDA"
    B, H, N, D = q.shape
    K_TOP = topk_indices.shape[-1]
    BH = B * H

    # collapse (B, H) -> single dim, all contiguous
    q_c = q.contiguous().view(BH, N, D)
    k_c = k.contiguous().view(BH, N, D)
    v_c = v.contiguous().view(BH, N, D)
    idx_c = topk_indices.contiguous().to(torch.int64).view(BH, N, K_TOP)
    out = torch.empty_like(q_c)

    BLOCK_K = 32
    if K_TOP < BLOCK_K:
        BLOCK_K = max(8, triton.next_power_of_2(K_TOP))

    # All tensors are (BH, N, *) in row-major, so stride along BH is the
    # whole-row size. We pass the same value as both batch-stride and
    # head-stride to keep the kernel signature symmetric.
    s_qbh = q_c.stride(0); s_qn = q_c.stride(1); s_qd = q_c.stride(2)
    s_kbh = k_c.stride(0); s_kn = k_c.stride(1); s_kd = k_c.stride(2)
    s_vbh = v_c.stride(0); s_vn = v_c.stride(1); s_vd = v_c.stride(2)
    s_ibh = idx_c.stride(0); s_in = idx_c.stride(1); s_ik = idx_c.stride(2)
    s_obh = out.stride(0); s_on = out.stride(1); s_od = out.stride(2)

    grid = (BH, triton.cdiv(N, block_m))
    _topk_attn_fwd_kernel[grid](
        q_c, k_c, v_c, idx_c, out,
        s_qbh, 0, s_qn, s_qd,
        s_kbh, 0, s_kn, s_kd,
        s_vbh, 0, s_vn, s_vd,
        s_ibh, 0, s_in, s_ik,
        s_obh, 0, s_on, s_od,
        N, K_TOP, 1.0 / math.sqrt(D),
        BLOCK_M=block_m, BLOCK_K=BLOCK_K, D=D,
        IS_CAUSAL=causal,
    )

    return out.view(B, H, N, D)
