"""Perplexity evaluation on a slice of WikiText-2.

Replaces the seven-paragraph reference text used everywhere else with a
real test set. Downloads the file lazily on first use.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional
import urllib.request

import torch

from ossa.bench.sparse_forward import (
    PatchPlan,
    _load_router,
    _patch_layer,
    compute_ppl,
)
from ossa.capture import load_model


WIKITEXT_URL = (
    "https://huggingface.co/datasets/wikitext/raw/main/wikitext-2-raw-v1/"
    "test-00000-of-00001.parquet"
)
WIKITEXT_FALLBACK_URL = (
    "https://raw.githubusercontent.com/openai/gpt-2-output-dataset/master/"
    "data/webtext.test.jsonl"
)
DATA_DIR = Path("bench/data")


def load_wikitext_text(seq_len: int = 1024, n_chunks: int = 4) -> str:
    """Fetch a chunk of WikiText-2 test split. Falls back to a small
    bundled corpus if the download fails.

    The function returns a single concatenated string. We deliberately
    avoid the parquet/HF datasets dependency: we fetch the raw text via
    a stable mirror.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "wikitext_test.txt"
    if not cache.exists():
        url = (
            "https://raw.githubusercontent.com/pytorch/examples/main/"
            "word_language_model/data/wikitext-2/test.txt"
        )
        print(f"[wikitext] downloading {url}")
        try:
            urllib.request.urlretrieve(url, cache)
        except Exception as exc:
            print(f"[wikitext] download failed: {exc}; using bundled fallback")
            cache.write_text(_BUNDLED_FALLBACK, encoding="utf-8")
    text = cache.read_text(encoding="utf-8")
    # Compress whitespace and trim
    text = " ".join(text.split())
    return text


_BUNDLED_FALLBACK = """
The history of natural language processing began in the 1950s with experiments
in machine translation. Early systems applied hand-written rules to map source
sentences to target ones, but the difficulty of disambiguation soon forced a
shift toward statistical methods. The introduction of large parallel corpora
in the 1990s, the rise of recurrent neural networks in the early 2010s, and
finally the transformer architecture in 2017 each transformed the field.
Modern language models trained on web-scale data routinely produce fluent
output across dozens of languages, though they remain unreliable on tasks
that require multi-step reasoning or precise factual recall.
""" * 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/content_router_layer14.pt"))
    parser.add_argument("--save", type=Path, default=Path("bench/results/wikitext_eval.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)

    text = load_wikitext_text(seq_len=args.seq_len)
    n_tokens = len(tokenizer.encode(text))
    print(f"[wikitext] usable tokens = {n_tokens} (truncated to {args.seq_len})")

    router = _load_router(args.checkpoint, model, args.layer, args.seq_len)

    plan = PatchPlan(mode="dense", k=args.ks[0], layers=())
    restore = _patch_layer(model, plan)
    ppl_dense = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
    restore()
    print(f"\ndense        ppl = {ppl_dense:.4f}")

    rows: list[dict] = []
    started = time.perf_counter()
    for k in args.ks:
        plan = PatchPlan(mode="oracle", k=k, layers=(args.layer,))
        restore = _patch_layer(model, plan)
        ppl_oracle = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        plan = PatchPlan(
            mode="router", k=k, layers=(args.layer,), routers={args.layer: router},
        )
        restore = _patch_layer(model, plan)
        ppl_router = compute_ppl(model, tokenizer, text, seq_len=args.seq_len)
        restore()

        d_oracle = ppl_oracle / ppl_dense - 1
        d_router = ppl_router / ppl_dense - 1
        print(
            f"  k={k:>4}: oracle={ppl_oracle:.3f} ({d_oracle*100:+.2f}%)   "
            f"router={ppl_router:.3f} ({d_router*100:+.2f}%)"
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
    print(f"  WikiText-2 eval — layer {args.layer}, seq={args.seq_len}, "
          f"elapsed {elapsed:.1f}s")
    print("=" * 72)
    print(f"{'k':>5}  {'ppl_dense':>10}  {'oracle %':>10}  {'router %':>10}")
    for r in rows:
        print(
            f"{r['k']:>5}  {r['ppl_dense']:>10.3f}  "
            f"{r['delta_oracle']*100:>+9.2f}%  "
            f"{r['delta_router']*100:>+9.2f}%"
        )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"rows": rows, "layer": args.layer}, indent=2))
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
