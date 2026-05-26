"""End-to-end perplexity with router-driven sparse attention on
multiple layers simultaneously.

Loads a list of trained ``content_router_layer{L}.pt`` checkpoints,
patches every one of those layers with the trained router, leaves the
remaining layers dense. Reports perplexity for each setting:

  * dense                       (baseline)
  * oracle every patched layer  (upper bound)
  * router every patched layer  (the publishable number)

This is the multi-layer extension of ``router_sweep.py``.
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
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layers to patch (must each have a checkpoint).")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save", type=Path, default=Path("bench/results/multi_layer_ppl.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)
    text = _build_text(args.seq_len, tokenizer)
    n_tokens = len(tokenizer.encode(text))
    print(f"text: {n_tokens} tokens (truncated to {args.seq_len})")

    # Load every checkpoint
    routers: dict[int, object] = {}
    for layer in args.layers:
        ckpt = args.checkpoint_dir / f"content_router_layer{layer}.pt"
        if not ckpt.exists():
            print(f"  skip layer {layer}: {ckpt} missing")
            continue
        routers[layer] = _load_router(ckpt, model, layer, args.seq_len)
        print(f"  loaded router for layer {layer}")

    if not routers:
        print("no checkpoints found, abort")
        return

    available_layers = tuple(sorted(routers.keys()))

    # Dense baseline
    plan = PatchPlan(mode="dense", k=args.ks[0], layers=())
    restore = _patch_layer(model, plan)
    ppl_dense = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
    restore()
    print(f"\ndense          ppl = {ppl_dense:.4f}")

    rows: list[dict] = []
    started = time.perf_counter()
    for k in args.ks:
        # Oracle on those layers
        plan = PatchPlan(mode="oracle", k=k, layers=available_layers)
        restore = _patch_layer(model, plan)
        ppl_oracle = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        # Router on those layers
        plan = PatchPlan(mode="router", k=k, layers=available_layers, routers=routers)
        restore = _patch_layer(model, plan)
        ppl_router = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        d_oracle = ppl_oracle / ppl_dense - 1
        d_router = ppl_router / ppl_dense - 1
        gap = ppl_router - ppl_oracle
        print(
            f"  k={k:>4}: oracle={ppl_oracle:.3f} ({d_oracle*100:+.2f}%)   "
            f"router={ppl_router:.3f} ({d_router*100:+.2f}%)   gap={gap:+.3f}"
        )
        rows.append({
            "k": k,
            "ppl_dense": ppl_dense,
            "ppl_oracle": ppl_oracle,
            "ppl_router": ppl_router,
            "delta_oracle": d_oracle,
            "delta_router": d_router,
        })

    elapsed = time.perf_counter() - started

    print()
    print("=" * 72)
    print(f"  Multi-layer PPL — patched layers {list(available_layers)}, "
          f"seq_len={args.seq_len}, elapsed {elapsed:.1f}s")
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
    args.save.write_text(json.dumps(
        {"layers": list(available_layers), "rows": rows}, indent=2,
    ))
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
