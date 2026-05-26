"""Train a ContentRouter that imitates dense attention on one layer.

Mirror of ``full_train.py`` but using ``ContentRouter`` (which sees the
real keys of the wrapped layer) and capturing both q_proj and k_proj
via a forward hook.

If recall@k clears 0.7 on a held-out prompt set we are clear to plug
the router into ``sparse_forward.py`` and read the perplexity number.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ossa.bench.full_train import PROMPTS
from ossa.capture import capture_attention_scores, load_model
from ossa.router import ContentRouter, ContentRouterConfig, distillation_loss


@dataclass
class TrainConfig:
    layer: int = 14
    seq_len: int = 512
    k: int = 64
    proj_dim: int = 64
    steps: int = 1500
    lr: float = 1e-3
    log_every: int = 100
    holdout_size: int = 5
    seed: int = 42


def _ensure_seq_len(text: str, tokenizer, seq_len: int) -> str:
    while len(tokenizer.encode(text)) < seq_len + 8:
        text = text + " " + text
    return text


def _capture_qk_dense(
    model, tokenizer, text: str, *, seq_len: int, layer: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (q_real (1,H,N,head_dim), k_real (1,H,N,head_dim),
    dense_attn (1,H,N,N))."""

    text = _ensure_seq_len(text, tokenizer, seq_len)
    captured = capture_attention_scores(model, tokenizer, text, seq_len=seq_len)
    dense = captured["layers"][layer].to(model.device)
    ids_dev = captured["ids"].to(model.device)

    holder: dict[str, torch.Tensor] = {}

    def hook(module, args, kwargs, output):  # noqa: ANN001
        hidden = args[0] if args else kwargs["hidden_states"]
        Bsz, N, _ = hidden.shape
        head_dim = (
            module.head_dim
            if hasattr(module, "head_dim")
            else module.config.hidden_size // module.config.num_attention_heads
        )
        n_heads = module.config.num_attention_heads
        n_kv_heads = module.config.num_key_value_heads

        q_proj = module.q_proj(hidden).view(Bsz, N, n_heads, head_dim).permute(0, 2, 1, 3)
        k_proj = module.k_proj(hidden).view(Bsz, N, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        # repeat KV heads to match Q heads (GQA)
        rep = n_heads // n_kv_heads
        if rep > 1:
            k_proj = k_proj.repeat_interleave(rep, dim=1)
        holder["q"] = q_proj.float().detach().contiguous()
        holder["k"] = k_proj.float().detach().contiguous()

    layer_module = model.model.layers[layer].self_attn
    handle = layer_module.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=ids_dev)
    finally:
        handle.remove()

    return holder["q"], holder["k"], dense


def _recall_topk(
    router_logits: torch.Tensor, dense: torch.Tensor, *, k: int, sample_n: int = 64
) -> float:
    B, H, N, _ = dense.shape
    rng = torch.Generator(device="cpu").manual_seed(0)
    n_idx = torch.randperm(N, generator=rng)[:sample_n]
    overlaps: list[float] = []
    with torch.no_grad():
        d_top = torch.topk(dense[:, :, n_idx, :], k=k, dim=-1).indices
        r_top = torch.topk(router_logits[:, :, n_idx, :], k=k, dim=-1).indices
        for b in range(B):
            for h in range(H):
                for q_i in range(sample_n):
                    d_set = set(d_top[b, h, q_i].tolist())
                    r_set = set(r_top[b, h, q_i].tolist())
                    overlaps.append(len(d_set & r_set) / k)
    return float(np.mean(overlaps))


def run(cfg: TrainConfig, *, model_name: str, save_dir: Path, device: Optional[str] = None) -> dict:
    started = time.perf_counter()
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)

    rng = np.random.default_rng(cfg.seed)
    indices = list(range(len(PROMPTS)))
    rng.shuffle(indices)
    holdout = [PROMPTS[i] for i in indices[: cfg.holdout_size]]
    train_pool = [PROMPTS[i] for i in indices[cfg.holdout_size :]]

    q0, k0, _ = _capture_qk_dense(model, tokenizer, train_pool[0], seq_len=cfg.seq_len, layer=cfg.layer)
    n_heads, head_dim = q0.shape[1], q0.shape[3]
    print(f"layer {cfg.layer}: n_heads={n_heads}, head_dim={head_dim}")

    router_cfg = ContentRouterConfig(
        head_dim=head_dim, n_heads=n_heads, seq_len=cfg.seq_len, proj_dim=cfg.proj_dim,
    )
    router = ContentRouter(router_cfg).to(model.device)
    optim = torch.optim.AdamW(router.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"router params: {n_params:,}")

    log = {
        "config": vars(cfg),
        "n_params": n_params,
        "history": [],
    }

    pool_n = len(train_pool)
    print(f"training {cfg.steps} steps on {pool_n} prompts, k={cfg.k} ...")
    for step in range(cfg.steps):
        prompt = train_pool[step % pool_n]
        q, k, dense = _capture_qk_dense(model, tokenizer, prompt, seq_len=cfg.seq_len, layer=cfg.layer)
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
                    q_h, k_h, d_h = _capture_qk_dense(model, tokenizer, hp, seq_len=cfg.seq_len, layer=cfg.layer)
                    logits_h = router.score(q_h, k_h)
                    recalls.append(_recall_topk(logits_h, d_h, k=cfg.k))
            mean_recall = float(np.mean(recalls))
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "holdout_recall": mean_recall,
            }
            log["history"].append(row)
            print(
                f"  step {step+1:>4d}: train_loss={loss.item():.4f}  "
                f"holdout recall@{cfg.k}={mean_recall:.3f}"
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": router.state_dict(), "cfg": vars(cfg)},
        save_dir / f"content_router_layer{cfg.layer}.pt",
    )
    log["seconds"] = time.perf_counter() - started
    log["final_recall"] = log["history"][-1]["holdout_recall"]
    log["final_loss"] = log["history"][-1]["loss"]
    (save_dir / f"content_log_layer{cfg.layer}.json").write_text(json.dumps(log, indent=2))

    print()
    print(f"final loss          = {log['final_loss']:.4f}")
    print(f"final holdout recall= {log['final_recall']:.3f}")
    print(f"elapsed             = {log['seconds']:.1f}s")

    if log["final_recall"] >= 0.70:
        print("VERDICT: STRONG -- content router generalises. Run sparse_forward with checkpoint.")
    elif log["final_recall"] >= 0.50:
        print("VERDICT: MEDIUM -- usable signal. Try larger proj_dim or more steps.")
    else:
        print("VERDICT: WEAK -- content router still not learning the pattern.")

    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    cfg = TrainConfig(
        layer=args.layer, seq_len=args.seq_len, k=args.k, proj_dim=args.proj_dim,
        steps=args.steps, lr=args.lr, log_every=args.log_every,
    )
    run(cfg, model_name=args.model, save_dir=args.save_dir, device=args.device)


if __name__ == "__main__":
    main()
