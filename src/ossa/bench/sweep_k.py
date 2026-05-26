"""Sweep top-K and sequence length for the oracle sparse forward.

Answers: at what k does the model start losing quality? Does it depend
on sequence length? This calibrates how aggressive a router can afford
to be before quality breaks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch

from ossa.bench.sparse_forward import (
    PatchPlan,
    _build_text,
    _patch_layer,
    compute_ppl,
)
from ossa.capture import load_model


def run(
    *,
    model_name: str,
    seq_lens: tuple[int, ...],
    ks: tuple[int, ...],
    save: Optional[Path],
    device: Optional[str],
) -> dict:
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)
    n_layers = model.config.num_hidden_layers
    layers = tuple(range(n_layers))

    rows: list[dict] = []
    started = time.perf_counter()

    for seq_len in seq_lens:
        text = _build_text(seq_len, tokenizer)
        n_tokens = len(tokenizer.encode(text))
        print(f"\nseq_len={seq_len} (text has {n_tokens} tokens)")

        # baseline once per seq_len
        plan = PatchPlan(mode="dense", k=ks[0], layers=())
        restore = _patch_layer(model, plan)
        ppl_dense = compute_ppl(model, tokenizer, text, seq_len=seq_len)
        restore()
        print(f"  dense       ppl = {ppl_dense:.3f}")

        for k in ks:
            ratio = k / seq_len
            plan = PatchPlan(mode="oracle", k=k, layers=layers)
            restore = _patch_layer(model, plan)
            ppl_oracle = compute_ppl(model, tokenizer, text, seq_len=seq_len)
            restore()
            delta = ppl_oracle / ppl_dense - 1
            print(
                f"  oracle k={k:>4d} ({ratio:>6.1%}): "
                f"ppl={ppl_oracle:.3f}  delta={delta:+.1%}"
            )
            rows.append({
                "seq_len": seq_len,
                "k": k,
                "k_ratio": ratio,
                "ppl_dense": ppl_dense,
                "ppl_oracle": ppl_oracle,
                "delta": delta,
            })

    elapsed = time.perf_counter() - started

    # Print summary
    print("\n" + "=" * 78)
    print("  Oracle sparse forward — perplexity penalty by k and seq_len")
    print("=" * 78)
    seq_set = sorted({r["seq_len"] for r in rows})
    k_set = sorted({r["k"] for r in rows})
    header = f"{'k':>5}   " + "   ".join(f"seq{sl}".rjust(10) for sl in seq_set)
    print(header)
    print("-" * len(header))
    for k in k_set:
        cells: list[str] = []
        for sl in seq_set:
            row = next(r for r in rows if r["k"] == k and r["seq_len"] == sl)
            cells.append(f"{row['delta']*100:+8.2f}%")
        print(f"{k:>5d}   " + "   ".join(c.rjust(10) for c in cells))

    out = {"model": model_name, "rows": rows, "elapsed_s": elapsed}
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {save}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--save", type=Path, default=Path("bench/results/sweep_k.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(
        model_name=args.model,
        seq_lens=tuple(args.seq_lens),
        ks=tuple(args.ks),
        save=args.save,
        device=args.device,
    )


if __name__ == "__main__":
    main()
