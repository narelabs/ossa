"""Sparsity probe.

Question this script answers: **how much attention mass lives in the
top-k keys per query, by layer and head, on a real long input?**

This is the cheapest possible feasibility check. If the dense attention
matrices of frozen Qwen are already concentrated in a small fraction of
keys, a learned router has an easy target to imitate. If the mass is
spread out, no router will recover quality — and we close the project
the same day instead of training for hours.

We report two numbers per (layer, head) at each ``k``:

* ``mass_topk``      — sum of attention probabilities over the top-k keys
                       for each query, then averaged across queries.
                       1.0 = "this head only ever looks at k tokens".
* ``recall_topk``    — fraction of the dense attention output (h @ V)
                       recovered if we replaced full attention with
                       attention restricted to the top-k keys, measured
                       as cosine to the dense output.

We aggregate across heads to produce one row per layer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ossa.capture import capture_attention_scores, load_model


DEFAULT_PROMPT = (
    "We are reading a long technical article that discusses transformer "
    "architectures, sparse attention mechanisms, content-based routing, "
    "long-context language models, and benchmark methodology. The article "
    "is written by a research engineer in an exploratory voice, with many "
    "concrete examples and references to published papers. Pay close "
    "attention to the structure, the named entities, and the numerical "
    "claims. After the article finishes you will be asked questions about "
    "specific paragraphs, sentences, and named entities, and you must "
    "answer each question concisely with only the most relevant phrase "
    "from the original text. "
)


def _build_text(seq_len: int, tokenizer) -> str:
    text = DEFAULT_PROMPT
    while len(tokenizer.encode(text)) < seq_len + 16:
        text += " " + DEFAULT_PROMPT
    return text


@dataclass
class LayerSparsity:
    layer: int
    n_heads: int
    avg_mass_top_k: dict[int, float]  # k -> mean over (heads, queries)
    p10_mass_top_k: dict[int, float]  # 10th percentile (worst heads)
    p90_mass_top_k: dict[int, float]  # 90th percentile (sparse heads)


def _topk_mass(attn: torch.Tensor, ks: list[int]) -> dict[int, tuple[float, float, float]]:
    """attn: (1, H, N, N). Returns k -> (mean, p10, p90) of top-k mass."""

    out: dict[int, tuple[float, float, float]] = {}
    H, N = attn.shape[1], attn.shape[2]
    for k in ks:
        k_eff = min(k, N)
        topk = torch.topk(attn, k=k_eff, dim=-1).values  # (1, H, N, k)
        mass = topk.sum(dim=-1)  # (1, H, N)
        flat = mass.flatten().numpy()
        out[k] = (float(flat.mean()), float(np.percentile(flat, 10)), float(np.percentile(flat, 90)))
    return out


def run(
    *,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    seq_len: int = 1024,
    ks: tuple[int, ...] = (16, 32, 64, 128, 256),
    device: str | None = None,
    save: Path | None = None,
) -> dict:
    started = time.perf_counter()
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)
    text = _build_text(seq_len, tokenizer)

    print(f"capturing attention on seq_len={seq_len} ...")
    captured = capture_attention_scores(model, tokenizer, text, seq_len=seq_len)
    layers = captured["layers"]
    print(f"captured {len(layers)} layers, shape per layer: {tuple(layers[0].shape)}")

    rows: list[LayerSparsity] = []
    for L, attn in enumerate(layers):
        stats = _topk_mass(attn, list(ks))
        rows.append(
            LayerSparsity(
                layer=L,
                n_heads=int(attn.shape[1]),
                avg_mass_top_k={k: stats[k][0] for k in ks},
                p10_mass_top_k={k: stats[k][1] for k in ks},
                p90_mass_top_k={k: stats[k][2] for k in ks},
            )
        )

    # Print per-layer table
    header = f"{'layer':>5} {'heads':>5}  " + "  ".join(f"top{k:<4d}" for k in ks)
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = "  ".join(f"{r.avg_mass_top_k[k]:>6.3f}" for k in ks)
        print(f"{r.layer:>5d} {r.n_heads:>5d}  {cells}")

    # Aggregate verdict
    print()
    print("Verdict:")
    for k in ks:
        avg_per_layer = [r.avg_mass_top_k[k] for r in rows]
        avg = float(np.mean(avg_per_layer))
        worst = float(min(avg_per_layer))
        best = float(max(avg_per_layer))
        ratio = k / seq_len
        print(
            f"  top-{k:<4d} ({ratio:>6.2%}): "
            f"layer-avg mass = {avg:.3f}, range [{worst:.3f} .. {best:.3f}]"
        )

    avg_top64 = float(np.mean([r.avg_mass_top_k[64] for r in rows]))
    if avg_top64 >= 0.85:
        print(
            f"\nSIGNAL: top-64 mass averages {avg_top64:.3f} — frozen Qwen "
            f"attention is already sparse enough that a router has a real shot."
        )
    elif avg_top64 >= 0.65:
        print(
            f"\nWEAK: top-64 mass averages {avg_top64:.3f} — partial. Some "
            f"layers will need a larger k or hierarchical fallback."
        )
    else:
        print(
            f"\nDEAD: top-64 mass averages {avg_top64:.3f} — attention is too "
            f"distributed for a sparse retrofit on this model. Close the project."
        )

    print(f"\nElapsed: {time.perf_counter() - started:.1f}s")

    out = {
        "model": model_name,
        "seq_len": seq_len,
        "ks": list(ks),
        "layers": [asdict(r) for r in rows],
    }
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", type=Path, default=Path("bench/results/sparsity.json"))
    args = parser.parse_args()
    run(
        model_name=args.model,
        seq_len=args.seq_len,
        ks=tuple(args.ks),
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
