"""Wall-clock benchmark: dense attention vs true sparse forward.

Measures forward time for one transformer layer's attention on the same
(Q, K, V) tensors. Compares:

  - dense attention (matmul Q·K^T, softmax, A·V)
  - sparse_attention_forward_chunked with oracle top-K indices (the
    speed of any actual sparse-attention scheme)

This is the number that says whether OSSA gives speed in addition to
quality. The mask-mode forward used in ``sparse_forward.py`` for
perplexity evaluation is **not** here on purpose: it has dense
compute by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ossa.sparse_attention import (
    dense_attention_reference,
    sparse_attention_forward_chunked,
)


@torch.no_grad()
def bench_one(
    *,
    seq_len: int,
    k: int,
    n_heads: int = 12,
    head_dim: int = 128,
    n_warmup: int = 3,
    n_iter: int = 10,
    device: torch.device,
) -> dict:
    q = torch.randn(1, n_heads, seq_len, head_dim, device=device)
    kk = torch.randn(1, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(1, n_heads, seq_len, head_dim, device=device)

    # Pre-compute "router output" = top-K of dense scores. This is what
    # a router would supply at inference. Cost of running the router
    # itself is small relative to attention and we measure attention only.
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ kk.transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    topk_idx = scores.topk(min(k, seq_len), dim=-1).indices

    # ----- dense -----
    for _ in range(n_warmup):
        _ = dense_attention_reference(q, kk, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_dense = dense_attention_reference(q, kk, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dense_ms = (time.perf_counter() - t0) / n_iter * 1000

    # ----- sparse -----
    for _ in range(n_warmup):
        _ = sparse_attention_forward_chunked(q, kk, v, topk_indices=topk_idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_sparse = sparse_attention_forward_chunked(q, kk, v, topk_indices=topk_idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sparse_ms = (time.perf_counter() - t0) / n_iter * 1000

    # quality check: oracle top-K should reproduce dense well
    cos = torch.nn.functional.cosine_similarity(
        out_dense.flatten(end_dim=-2), out_sparse.flatten(end_dim=-2), dim=-1
    ).mean().item()

    return {
        "seq_len": seq_len,
        "k": k,
        "k_ratio": k / seq_len,
        "dense_ms": round(dense_ms, 3),
        "sparse_ms": round(sparse_ms, 3),
        "speedup": round(dense_ms / max(sparse_ms, 1e-9), 2),
        "output_cosine": round(cos, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", type=Path, default=Path("bench/results/wallclock.json"))
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}, n_heads={args.n_heads}, head_dim={args.head_dim}")

    rows: list[dict] = []
    for sl in args.seq_lens:
        print(f"\nseq_len={sl}")
        for k in args.ks:
            if k >= sl:
                continue
            row = bench_one(
                seq_len=sl, k=k,
                n_heads=args.n_heads, head_dim=args.head_dim,
                device=device,
            )
            print(
                f"  k={k:>4} ({row['k_ratio']:>6.1%}): "
                f"dense {row['dense_ms']:>7.2f}ms  "
                f"sparse {row['sparse_ms']:>7.2f}ms  "
                f"speedup {row['speedup']:>5.2f}x  "
                f"cos {row['output_cosine']:.3f}"
            )
            rows.append(row)

    print()
    print("=" * 78)
    print("  Wall-clock summary")
    print("=" * 78)
    print(f"{'seq_len':>8}  {'k':>5}  {'k/N':>6}  {'dense ms':>10}  {'sparse ms':>10}  {'speedup':>8}")
    for r in rows:
        print(
            f"{r['seq_len']:>8}  {r['k']:>5}  {r['k_ratio']:>5.1%}  "
            f"{r['dense_ms']:>10.2f}  {r['sparse_ms']:>10.2f}  {r['speedup']:>7.2f}x"
        )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
