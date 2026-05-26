"""Train ContentRouter on several layers in one process.

Loads the model once, trains a router per layer in a list, saves
each checkpoint and per-layer log. Designed to validate that the
router pattern observed on layer 14 generalises across the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ossa.bench.content_train import (
    TrainConfig,
    _capture_qk_dense,
    _recall_topk,
)
from ossa.bench.full_train import PROMPTS
from ossa.capture import load_model
from ossa.router import ContentRouter, ContentRouterConfig, distillation_loss


def train_one_layer(
    model, tokenizer, *, layer: int, cfg: TrainConfig, save_dir: Path,
) -> dict:
    rng = np.random.default_rng(cfg.seed)
    indices = list(range(len(PROMPTS)))
    rng.shuffle(indices)
    holdout = [PROMPTS[i] for i in indices[: cfg.holdout_size]]
    train_pool = [PROMPTS[i] for i in indices[cfg.holdout_size :]]

    q0, k0, _ = _capture_qk_dense(
        model, tokenizer, train_pool[0], seq_len=cfg.seq_len, layer=layer
    )
    n_heads, head_dim = q0.shape[1], q0.shape[3]
    print(f"\n[layer {layer}] n_heads={n_heads}, head_dim={head_dim}")

    router_cfg = ContentRouterConfig(
        head_dim=head_dim, n_heads=n_heads, seq_len=cfg.seq_len, proj_dim=cfg.proj_dim,
    )
    router = ContentRouter(router_cfg).to(model.device)
    optim = torch.optim.AdamW(router.parameters(), lr=cfg.lr)

    history: list[dict] = []
    pool_n = len(train_pool)
    started = time.perf_counter()

    for step in range(cfg.steps):
        prompt = train_pool[step % pool_n]
        q, k, dense = _capture_qk_dense(
            model, tokenizer, prompt, seq_len=cfg.seq_len, layer=layer
        )
        router.train()
        logits = router.score(q, k)
        loss = distillation_loss(logits, dense, k=cfg.k)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step == 0 or (step + 1) % cfg.log_every == 0:
            router.eval()
            recalls: list[float] = []
            with torch.no_grad():
                for hp in holdout:
                    q_h, k_h, d_h = _capture_qk_dense(
                        model, tokenizer, hp, seq_len=cfg.seq_len, layer=layer,
                    )
                    logits_h = router.score(q_h, k_h)
                    recalls.append(_recall_topk(logits_h, d_h, k=cfg.k))
            mean_recall = float(np.mean(recalls))
            history.append({"step": step + 1, "loss": float(loss.item()),
                            "holdout_recall": mean_recall})
            print(
                f"  [layer {layer}] step {step+1:>4d}: loss={loss.item():.4f}  "
                f"recall@{cfg.k}={mean_recall:.3f}"
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": router.state_dict(), "cfg": vars(cfg)},
        save_dir / f"content_router_layer{layer}.pt",
    )
    elapsed = time.perf_counter() - started
    log = {
        "layer": layer,
        "config": vars(cfg),
        "history": history,
        "final_loss": history[-1]["loss"],
        "final_recall": history[-1]["holdout_recall"],
        "elapsed_s": elapsed,
    }
    (save_dir / f"content_log_layer{layer}.json").write_text(json.dumps(log, indent=2))
    print(f"[layer {layer}] done in {elapsed:.1f}s, final recall={log['final_recall']:.3f}")
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 21, 27])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)

    summary: list[dict] = []
    for layer in args.layers:
        cfg = TrainConfig(
            layer=layer, seq_len=args.seq_len, k=args.k, proj_dim=args.proj_dim,
            steps=args.steps, lr=args.lr, log_every=args.log_every,
        )
        log = train_one_layer(model, tokenizer, layer=layer, cfg=cfg, save_dir=args.save_dir)
        summary.append({
            "layer": layer,
            "final_recall": log["final_recall"],
            "final_loss": log["final_loss"],
            "elapsed_s": log["elapsed_s"],
        })

    print()
    print("=" * 60)
    print("  Multi-layer training summary")
    print("=" * 60)
    print(f"{'layer':>5}  {'recall':>8}  {'loss':>8}  {'time s':>8}")
    for r in summary:
        print(
            f"{r['layer']:>5}  {r['final_recall']:>8.3f}  "
            f"{r['final_loss']:>8.4f}  {r['elapsed_s']:>8.1f}"
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "multi_layer_summary.json").write_text(
        json.dumps({"layers": summary}, indent=2)
    )


if __name__ == "__main__":
    main()
