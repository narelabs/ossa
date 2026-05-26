"""Micro-train sanity check for the OSSA router.

Run for ~50 steps. If the distillation loss does not decrease meaningfully
(say, less than 30 % reduction from step 0 to step 50), the router
architecture is broken and a 4-hour training run would just waste GPU.

This is an honest "is the loss landscape friendly?" check, not the real
training. The real training uses many more samples, larger sequences, and
warm-up / scheduling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ossa.bench.sparsity import _build_text
from ossa.capture import capture_attention_scores, load_model
from ossa.router import PerLayerRouter, RouterConfig, distillation_loss, router_recall_at_k


def run(
    *,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    layer: int = 14,
    seq_len: int = 512,
    k: int = 64,
    steps: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
    save: Path | None = None,
) -> dict:
    started = time.perf_counter()
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)
    text = _build_text(seq_len, tokenizer)

    print(f"capturing dense attention at layer {layer} on seq_len={seq_len} ...")
    captured = capture_attention_scores(model, tokenizer, text, seq_len=seq_len)
    dense_full = captured["layers"][layer]  # (1, H, N, N)

    # Capture the *real* query projection that the layer used. This means
    # one extra forward pass with a hook. Worth it: training the router on
    # a stand-in (token embeddings) is what gave recall@k = 0 in the first
    # micro-run.
    q_real_holder: dict[str, torch.Tensor] = {}

    def q_hook(module, args, kwargs, output):  # noqa: ANN001
        hidden = args[0] if args else kwargs["hidden_states"]
        Bsz, N, _ = hidden.shape
        head_dim_real = (
            module.head_dim
            if hasattr(module, "head_dim")
            else module.config.hidden_size // module.config.num_attention_heads
        )
        q_proj = module.q_proj(hidden)  # (1, N, n_heads * head_dim_real)
        n_heads_real = q_proj.shape[-1] // head_dim_real
        q_proj = q_proj.view(Bsz, N, n_heads_real, head_dim_real).permute(0, 2, 1, 3).contiguous()
        q_real_holder["q"] = q_proj.float().detach()

    layer_module = model.model.layers[layer].self_attn
    handle = layer_module.register_forward_hook(q_hook, with_kwargs=True)
    try:
        ids_dev = captured["ids"].to(model.device)
        with torch.no_grad():
            model(input_ids=ids_dev)
    finally:
        handle.remove()

    if "q" not in q_real_holder:
        raise RuntimeError("q hook did not fire")
    q_input = q_real_holder["q"]
    n_heads = q_input.shape[1]
    head_dim = q_input.shape[3]
    print(f"real q_proj captured: shape={tuple(q_input.shape)}")

    cfg = RouterConfig(head_dim=head_dim, n_heads=n_heads, seq_len=seq_len)
    router = PerLayerRouter(cfg).to(model.device)
    router.train()

    optim = torch.optim.AdamW(router.parameters(), lr=lr)

    losses: list[float] = []
    print(f"micro-training {steps} steps, k={k} ...")
    for step in range(steps):
        logits = router(q_input)
        loss = distillation_loss(logits, dense_full.to(model.device), k=k)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  step {step+1:>3d}: loss = {loss.item():.4f}")

    router.eval()
    with torch.no_grad():
        final_logits = router(q_input)
        recall = router_recall_at_k(final_logits, dense_full.to(model.device), k=k)

    elapsed = time.perf_counter() - started
    delta = (losses[0] - losses[-1]) / max(losses[0], 1e-9)

    print()
    print(f"loss at step 0   : {losses[0]:.4f}")
    print(f"loss at step {steps:<3d}: {losses[-1]:.4f}")
    print(f"relative drop    : {delta:.1%}")
    print(f"recall@k (50%)   : {recall:.3f}")
    print(f"elapsed          : {elapsed:.1f}s")

    if delta >= 0.30 and recall >= 0.30:
        print("VERDICT: signal -- router is learning, full training is worth running.")
    elif delta >= 0.10:
        print("VERDICT: weak -- some learning, may need a better q_input or longer run.")
    else:
        print("VERDICT: dead -- loss landscape is bad. Try real q_proj or different arch.")

    out = {
        "model": model_name,
        "layer": layer,
        "seq_len": seq_len,
        "k": k,
        "steps": steps,
        "lr": lr,
        "losses": losses,
        "final_recall": recall,
        "rel_drop": delta,
        "elapsed_s": elapsed,
    }
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", type=Path, default=Path("bench/results/micro_train.json"))
    args = parser.parse_args()
    run(
        model_name=args.model,
        layer=args.layer,
        seq_len=args.seq_len,
        k=args.k,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
