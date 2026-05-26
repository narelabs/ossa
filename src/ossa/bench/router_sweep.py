"""Sweep K with the trained content router on one layer.

Loads the model and the router checkpoint **once**, then runs
dense + oracle + router perplexity for every k. Single-load is
important on Windows where reloading fp32 Qwen-1.5B several times
exceeds the paging file (OSError 1455).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ossa.bench.sparse_forward import (
    PatchPlan,
    _build_text,
    _load_router,
    _patch_layer,
    compute_ppl,
)
from ossa.capture import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/content_router_layer14.pt"),
    )
    parser.add_argument("--save", type=Path, default=Path("bench/results/router_sweep.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)
    text = _build_text(args.seq_len, tokenizer)

    print(f"loading router from {args.checkpoint} ...")
    router = _load_router(args.checkpoint, model, args.layer, args.seq_len)
    routers = {args.layer: router}

    # Dense baseline (one shot, mode=dense ignores layers)
    plan = PatchPlan(mode="dense", k=8, layers=())
    restore = _patch_layer(model, plan)
    ppl_dense = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
    restore()
    print(f"dense ppl = {ppl_dense:.4f}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for k in args.ks:
        # Oracle
        plan = PatchPlan(mode="oracle", k=k, layers=(args.layer,))
        restore = _patch_layer(model, plan)
        ppl_oracle = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        # Router
        plan = PatchPlan(mode="router", k=k, layers=(args.layer,), routers=routers)
        restore = _patch_layer(model, plan)
        ppl_router = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        d_oracle = ppl_oracle / ppl_dense - 1
        d_router = ppl_router / ppl_dense - 1
        gap = ppl_router - ppl_oracle
        print(
            f"  k={k:>4}: oracle ppl={ppl_oracle:.3f} ({d_oracle*100:+.2f}%)   "
            f"router ppl={ppl_router:.3f} ({d_router*100:+.2f}%)   gap={gap:+.3f}"
        )
        rows.append({
            "k": k,
            "ppl_dense": ppl_dense,
            "ppl_oracle": ppl_oracle,
            "ppl_router": ppl_router,
            "delta_oracle": d_oracle,
            "delta_router": d_router,
        })

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 72)
    print(f"  Router sweep on layer {args.layer}, seq_len={args.seq_len}, "
          f"elapsed {elapsed:.1f}s")
    print("=" * 72)
    print(f"{'k':>5}  {'ppl_dense':>10}  {'oracle %':>10}  {'router %':>10}  {'gap':>8}")
    for r in rows:
        print(
            f"{r['k']:>5}  {r['ppl_dense']:>10.3f}  "
            f"{r['delta_oracle']*100:>+9.2f}%  "
            f"{r['delta_router']*100:>+9.2f}%  "
            f"{(r['ppl_router'] - r['ppl_oracle']):>+7.3f}"
        )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"layer": args.layer, "rows": rows}, indent=2))
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
