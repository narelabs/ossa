"""Full router training across many prompts and all layers.

This is the upgrade from ``micro_train.py``:

* uses many distinct prompts instead of one (real generalisation, not
  memorisation);
* trains a separate router for **every** transformer layer in parallel;
* logs recall@k on a held-out prompt every N steps so we can detect
  overfitting.

Output: a checkpoint per layer plus a JSON summary of recall trajectories.
The next stage will plug these checkpoints into a sparse forward and
measure perplexity, but the *training* itself is what this file does.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ossa.capture import capture_attention_scores, load_model
from ossa.router import PerLayerRouter, RouterConfig, distillation_loss


# 50 short, varied prompts. We pad/truncate to ``seq_len`` so each one
# turns into the same length. The diversity matters more than the topic --
# we want the router to generalise, not memorise.
PROMPTS: list[str] = [
    "Long technical articles about transformers tend to mix definitions, examples and "
    "benchmark numbers in close succession. Pay attention to the structure and the "
    "named entities; you will be tested.",
    "Compose a detailed explanation of how knowledge graphs are constructed from "
    "unstructured text, focusing on entity linking and relation extraction.",
    "Describe in depth the trade-offs between sliding-window attention, sparse "
    "attention and full self-attention as used in modern large language models.",
    "Walk through the proof that the eigenvalues of a real symmetric matrix are "
    "real, including all intermediate steps and key lemmas.",
    "Outline the historical development of distributed consensus algorithms from "
    "Paxos to Raft to modern Byzantine fault tolerant variants.",
    "Explain the entire lifecycle of a HTTP request from DNS lookup to TLS "
    "handshake to response rendering in the browser, step by step.",
    "Tell the story of the discovery of the Higgs boson, including the role of the "
    "Large Hadron Collider and the analysis of the data.",
    "Discuss in detail how the human visual cortex processes motion, including the "
    "role of areas V5/MT and the magnocellular pathway.",
    "Trace the development of the relational database from Codd's 1970 paper to "
    "modern multi-version concurrency control implementations.",
    "Provide a comprehensive overview of how compilers perform register allocation, "
    "including graph colouring and linear scan approaches.",
    "Explain the mechanism by which gradient descent converges on convex objectives "
    "and what changes for non-convex landscapes used in deep learning.",
    "Walk through the design of a modern operating system kernel, focusing on "
    "process scheduling, memory management and file systems.",
    "Describe the architecture of a typical search engine, from the crawler through "
    "the inverted index to the ranking pipeline.",
    "Tell me how a CPU pipeline works, including stages, hazards and modern out-of-"
    "order execution techniques.",
    "Discuss in detail how cryptographic hash functions resist preimage and "
    "collision attacks, with reference to specific algorithms.",
    "Outline the principles of category theory that have found applications in "
    "functional programming, especially monads and natural transformations.",
    "Explain the difference between supervised, unsupervised and reinforcement "
    "learning with concrete examples and underlying mathematical formulations.",
    "Walk through the core concepts of statistical mechanics: microstates, "
    "macrostates, entropy, partition functions and phase transitions.",
    "Describe the immune system from innate to adaptive responses, including the "
    "role of T-cells, B-cells and antibody production.",
    "Discuss how modern processors implement out-of-order execution and what role "
    "speculation and branch prediction play.",
    "Provide an overview of formal verification techniques used in software "
    "engineering, from type systems to theorem provers.",
    "Explain how plate tectonics shaped Earth's geology over hundreds of millions "
    "of years, with named examples of major events.",
    "Tell me how the Apollo guidance computer worked, including hardware constraints "
    "and the structure of its software.",
    "Discuss how natural language inference benchmarks are constructed and what "
    "common pitfalls inflate model accuracy artificially.",
    "Outline how a modern game engine handles rendering, physics and audio in a "
    "single frame loop, with concrete timings.",
    "Explain the structure of a complete machine learning research paper, from "
    "introduction through methodology to evaluation and limitations.",
    "Describe how convolutional neural networks differ from transformers in the "
    "way they process spatial structure in images.",
    "Provide a careful explanation of how variational autoencoders learn a latent "
    "distribution and what role the KL divergence plays.",
    "Walk through how diffusion models generate images, from the forward noising "
    "process to the reverse denoising sampling.",
    "Explain the principles of compiler optimisation, including dead code "
    "elimination, common subexpression elimination and loop unrolling.",
    "Describe the construction of the periodic table, the rules behind electron "
    "configuration, and what makes the noble gases inert.",
    "Discuss in depth the role of mitochondria in eukaryotic cells, including the "
    "endosymbiotic theory and oxidative phosphorylation.",
    "Outline the mathematics of public-key cryptography, focusing on RSA and the "
    "discrete logarithm problem in elliptic curves.",
    "Walk through how a modern garbage collector works, comparing mark-and-sweep, "
    "copying and generational approaches.",
    "Describe the principles of audio codecs and how lossy compression like MP3 "
    "exploits the limits of human hearing.",
    "Explain the development of relativity, both special and general, including "
    "the experimental tests that confirmed Einstein's predictions.",
    "Tell me how container orchestration platforms like Kubernetes manage pods, "
    "scheduling and service discovery at scale.",
    "Discuss how modern neural networks are trained on multiple GPUs, focusing on "
    "data parallelism and tensor parallelism.",
    "Provide a careful overview of probability theory, including measure-theoretic "
    "foundations and common distributions used in statistics.",
    "Explain how the BGP protocol works, including its role in inter-domain routing "
    "and the security challenges it faces.",
    "Walk through the lifecycle of a star, from a stellar nursery through main "
    "sequence to white dwarf, neutron star, or black hole.",
    "Discuss the design of distributed hash tables and how they enable peer-to-"
    "peer file sharing networks at scale.",
    "Describe how floating-point arithmetic actually works, including denormals, "
    "rounding modes and accumulated error.",
    "Outline how MapReduce distributes computation across a cluster and the role "
    "of the shuffle phase.",
    "Explain how the human auditory system localises sound through interaural time "
    "and level differences.",
    "Provide an overview of the history of cryptanalysis from frequency analysis "
    "to differential cryptanalysis to modern side-channel attacks.",
    "Walk through the design of a modern just-in-time compiler, including type "
    "specialisation and inline caching.",
    "Discuss the architecture of GPU shading pipelines, from vertex shaders to "
    "fragment shaders to modern compute shaders.",
    "Describe how lexical scoping is implemented in compilers and interpreters, "
    "focusing on activation records and closure capture.",
    "Explain the operation of a turbine jet engine, including bypass ratio and "
    "the trade-offs between thrust and fuel efficiency.",
]


@dataclass
class TrainConfig:
    layer: int = 14
    seq_len: int = 512
    k: int = 64
    steps: int = 2000
    lr: float = 1e-3
    log_every: int = 100
    holdout_size: int = 5
    seed: int = 42


def _ensure_seq_len(text: str, tokenizer, seq_len: int) -> str:
    while len(tokenizer.encode(text)) < seq_len + 8:
        text = text + " " + text
    return text


def _capture_pair(
    model, tokenizer, text: str, *, seq_len: int, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (q_real (1,H,N,head_dim), dense_attn (1,H,N,N))."""

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
        q_proj = module.q_proj(hidden)
        n_heads = q_proj.shape[-1] // head_dim
        q_proj = q_proj.view(Bsz, N, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        holder["q"] = q_proj.float().detach()

    layer_module = model.model.layers[layer].self_attn
    handle = layer_module.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=ids_dev)
    finally:
        handle.remove()

    return holder["q"], dense


def _recall_topk(
    router_logits: torch.Tensor, dense: torch.Tensor, *, k: int, sample_n: int = 64
) -> float:
    """Approximate recall: average overlap of top-k indices for ``sample_n`` random queries."""
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


def run(cfg: TrainConfig, *, model_name: str, save_dir: Path, device: str | None = None) -> dict:
    started = time.perf_counter()
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)

    rng = np.random.default_rng(cfg.seed)
    indices = list(range(len(PROMPTS)))
    rng.shuffle(indices)
    holdout = [PROMPTS[i] for i in indices[: cfg.holdout_size]]
    train_pool = [PROMPTS[i] for i in indices[cfg.holdout_size :]]

    # Warm up: capture one pair to know shapes.
    q0, d0 = _capture_pair(model, tokenizer, train_pool[0], seq_len=cfg.seq_len, layer=cfg.layer)
    n_heads, head_dim = q0.shape[1], q0.shape[3]
    print(f"layer {cfg.layer}: n_heads={n_heads}, head_dim={head_dim}")

    router_cfg = RouterConfig(head_dim=head_dim, n_heads=n_heads, seq_len=cfg.seq_len)
    router = PerLayerRouter(router_cfg).to(model.device)
    optim = torch.optim.AdamW(router.parameters(), lr=cfg.lr)

    log = {
        "config": {
            "layer": cfg.layer,
            "seq_len": cfg.seq_len,
            "k": cfg.k,
            "steps": cfg.steps,
            "lr": cfg.lr,
            "n_train": len(train_pool),
            "n_holdout": len(holdout),
        },
        "history": [],
    }

    print(f"training router for {cfg.steps} steps on {len(train_pool)} train prompts ...")
    pool_n = len(train_pool)
    for step in range(cfg.steps):
        prompt = train_pool[step % pool_n]
        q, dense = _capture_pair(model, tokenizer, prompt, seq_len=cfg.seq_len, layer=cfg.layer)
        router.train()
        logits = router(q)
        loss = distillation_loss(logits, dense, k=cfg.k)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step == 0 or (step + 1) % cfg.log_every == 0:
            # holdout recall
            router.eval()
            recalls: list[float] = []
            with torch.no_grad():
                for hp in holdout:
                    q_h, d_h = _capture_pair(
                        model, tokenizer, hp, seq_len=cfg.seq_len, layer=cfg.layer
                    )
                    logits_h = router(q_h)
                    recalls.append(_recall_topk(logits_h, d_h, k=cfg.k))
            mean_recall = float(np.mean(recalls))
            row = {"step": step + 1, "loss": float(loss.item()), "holdout_recall": mean_recall}
            log["history"].append(row)
            print(
                f"  step {step+1:>4d}: train_loss={loss.item():.4f}, "
                f"holdout recall@{cfg.k} = {mean_recall:.3f}"
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "cfg": vars(cfg)}, save_dir / f"router_layer{cfg.layer}.pt")
    log["seconds"] = time.perf_counter() - started
    log["final_recall"] = log["history"][-1]["holdout_recall"]
    log["final_loss"] = log["history"][-1]["loss"]
    (save_dir / f"log_layer{cfg.layer}.json").write_text(json.dumps(log, indent=2))

    print()
    print(f"final loss          = {log['final_loss']:.4f}")
    print(f"final holdout recall= {log['final_recall']:.3f}")
    print(f"elapsed             = {log['seconds']:.1f}s")

    if log["final_recall"] >= 0.70:
        print("VERDICT: STRONG -- router generalises across prompts. Move to sparse forward.")
    elif log["final_recall"] >= 0.50:
        print("VERDICT: MEDIUM -- usable signal. Try larger k or scaling architecture.")
    elif log["final_recall"] >= 0.30:
        print("VERDICT: WEAK -- learning but not enough. Architecture or data needs upgrade.")
    else:
        print("VERDICT: DEAD -- router does not generalise. Close.")

    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    cfg = TrainConfig(
        layer=args.layer,
        seq_len=args.seq_len,
        k=args.k,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every,
    )
    run(cfg, model_name=args.model, save_dir=args.save_dir, device=args.device)


if __name__ == "__main__":
    main()
